// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "colmap/controllers/mipmap_matching.h"

#include "colmap/feature/index.h"
#include "colmap/util/file.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/logging.h"
#include "colmap/util/threading.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <string>
#include <utility>

namespace colmap {
namespace {

constexpr int kMipMapDescriptorDim = 128;

struct DescriptorRecord {
  image_t image_id = kInvalidImageId;
  point2D_t point2D_idx = kInvalidPoint2DIdx;
};

NodeHashMap<image_t, std::string> ImageNameMap(
    const std::vector<MipMapImageData>& images) {
  NodeHashMap<image_t, std::string> image_names;
  image_names.reserve(images.size());
  for (const auto& image : images) {
    image_names.emplace(image.image_id, image.image_name);
  }
  return image_names;
}

NodeHashMap<image_t, size_t> NumDescriptorsMap(
    const std::vector<MipMapImageData>& images) {
  NodeHashMap<image_t, size_t> num_descriptors;
  num_descriptors.reserve(images.size());
  for (const auto& image : images) {
    num_descriptors.emplace(image.image_id,
                            static_cast<size_t>(image.descriptors.data.rows()));
  }
  return num_descriptors;
}

void WriteCsvCell(std::ostream& stream, const std::string& value) {
  const bool needs_quotes = value.find_first_of(",\"\n\r") != std::string::npos;
  if (!needs_quotes) {
    stream << value;
    return;
  }

  stream << '"';
  for (const char ch : value) {
    if (ch == '"') {
      stream << "\"\"";
    } else {
      stream << ch;
    }
  }
  stream << '"';
}

}  // namespace

bool MipMapMatchingOptions::Check() const {
  if (num_images <= 0) {
    LOG(ERROR) << "MipMapMatching.num_images must be positive.";
    return false;
  }
  if (num_nearest_neighbors <= 0) {
    LOG(ERROR) << "MipMapMatching.num_nearest_neighbors must be positive.";
    return false;
  }
  if (max_num_features == 0) {
    LOG(ERROR) << "MipMapMatching.max_num_features must be negative or "
                  "positive.";
    return false;
  }
  return true;
}

std::vector<MipMapImageData> ReadMipMapImageDataFromDatabase(
    const Database& database, const int max_num_features) {
  std::vector<Image> database_images = database.ReadAllImages();
  std::sort(database_images.begin(),
            database_images.end(),
            [](const Image& image1, const Image& image2) {
              return image1.ImageId() < image2.ImageId();
            });

  std::vector<MipMapImageData> images;
  images.reserve(database_images.size());

  for (const auto& database_image : database_images) {
    const image_t image_id = database_image.ImageId();
    if (!database.ExistsDescriptors(image_id)) {
      continue;
    }

    FeatureDescriptors descriptors = database.ReadDescriptors(image_id);
    if (descriptors.data.rows() == 0) {
      continue;
    }
    if (descriptors.type != FeatureExtractorType::SIFT) {
      LOG(WARNING) << "Skipping image " << database_image.Name()
                   << " because MipMap Stage01 M1 currently expects SIFT "
                      "descriptors.";
      continue;
    }
    if (descriptors.data.cols() != kMipMapDescriptorDim) {
      LOG(WARNING) << "Skipping image " << database_image.Name()
                   << " because descriptor dimension is "
                   << descriptors.data.cols() << " instead of "
                   << kMipMapDescriptorDim << ".";
      continue;
    }

    if (max_num_features > 0 && descriptors.data.rows() > max_num_features) {
      descriptors.data.conservativeResize(max_num_features,
                                          descriptors.data.cols());
    }

    images.push_back({image_id, database_image.Name(), std::move(descriptors)});
  }

  return images;
}

MipMapCandidateGraph CreateMipMapCandidateGraph(
    const MipMapMatchingOptions& options,
    const std::vector<MipMapImageData>& images) {
  THROW_CHECK(options.Check());

  MipMapCandidateGraph graph;
  graph.candidate_rows.reserve(images.size());

  if (images.empty()) {
    return graph;
  }

  size_t num_global_descriptors = 0;
  for (const auto& image : images) {
    THROW_CHECK_EQ(image.descriptors.type, FeatureExtractorType::SIFT);
    THROW_CHECK_EQ(image.descriptors.data.cols(), kMipMapDescriptorDim);
    num_global_descriptors += image.descriptors.data.rows();
  }

  if (num_global_descriptors == 0) {
    return graph;
  }

  FeatureDescriptorsFloatData global_descriptors(
      static_cast<Eigen::Index>(num_global_descriptors), kMipMapDescriptorDim);
  std::vector<DescriptorRecord> records;
  records.reserve(num_global_descriptors);

  Eigen::Index row = 0;
  for (const auto& image : images) {
    for (Eigen::Index i = 0; i < image.descriptors.data.rows(); ++i) {
      global_descriptors.row(row) = image.descriptors.data.row(i).cast<float>();
      records.push_back({image.image_id, static_cast<point2D_t>(i)});
      ++row;
    }
  }

  auto index = FeatureDescriptorIndex::Create(
      FeatureDescriptorIndex::Type::FAISS,
      GetEffectiveNumThreads(options.num_threads));
  index->Build(FeatureDescriptorsFloat(FeatureExtractorType::SIFT,
                                       std::move(global_descriptors)));

  const NodeHashMap<image_t, size_t> num_descriptors =
      NumDescriptorsMap(images);
  FlatHashSet<image_pair_t> image_pair_ids;

  for (const auto& query_image : images) {
    MipMapImageCandidateRow candidate_row;
    candidate_row.image_id = query_image.image_id;

    FlatHashMap<image_t, double> accumulated_scores;
    accumulated_scores.reserve(images.size());
    for (const auto& image : images) {
      accumulated_scores.emplace(image.image_id, 0.0);
    }

    Eigen::RowMajorMatrixXi indices;
    Eigen::RowMajorMatrixXf l2_dists;
    index->Search(options.num_nearest_neighbors,
                  query_image.descriptors.ToFloat(),
                  indices,
                  l2_dists);

    for (Eigen::Index query_idx = 0; query_idx < indices.rows(); ++query_idx) {
      for (Eigen::Index rank = 0; rank < indices.cols(); ++rank) {
        const int global_idx = indices(query_idx, rank);
        if (global_idx < 0 || global_idx >= static_cast<int>(records.size())) {
          continue;
        }
        const image_t candidate_image_id = records[global_idx].image_id;
        const double dist = static_cast<double>(l2_dists(query_idx, rank));
        accumulated_scores[candidate_image_id] += 1.0 / (1.0 + dist);
      }
    }

    candidate_row.candidates.reserve(accumulated_scores.size());
    for (const auto& [image_id, accumulated_score] : accumulated_scores) {
      const auto num_candidate_desc = num_descriptors.find(image_id);
      THROW_CHECK(num_candidate_desc != num_descriptors.end());
      const double query_norm =
          1.0 / std::sqrt(static_cast<double>(std::max<Eigen::Index>(
                    1, query_image.descriptors.data.rows())));
      const double candidate_norm =
          1.0 / std::sqrt(static_cast<double>(
                    std::max<size_t>(1, num_candidate_desc->second)));
      double score = accumulated_score * query_norm * candidate_norm;
      if (image_id == query_image.image_id) {
        score = std::numeric_limits<double>::infinity();
      }
      candidate_row.candidates.push_back({image_id, score});
    }

    std::sort(candidate_row.candidates.begin(),
              candidate_row.candidates.end(),
              [](const MipMapImageCandidate& candidate1,
                 const MipMapImageCandidate& candidate2) {
                if (candidate1.score != candidate2.score) {
                  return candidate1.score > candidate2.score;
                }
                return candidate1.image_id < candidate2.image_id;
              });

    if (candidate_row.candidates.size() >
        static_cast<size_t>(options.num_images)) {
      candidate_row.candidates.resize(options.num_images);
    }

    for (const auto& candidate : candidate_row.candidates) {
      if (candidate.image_id == query_image.image_id) {
        continue;
      }
      const image_pair_t pair_id =
          ImagePairToPairId(query_image.image_id, candidate.image_id);
      if (image_pair_ids.insert(pair_id).second) {
        graph.image_pairs.push_back(PairIdToImagePair(pair_id));
      }
    }

    graph.candidate_rows.push_back(std::move(candidate_row));
  }

  return graph;
}

void WriteMipMapPairsText(const std::filesystem::path& path,
                          const MipMapCandidateGraph& graph,
                          const std::vector<MipMapImageData>& images) {
  const NodeHashMap<image_t, std::string> image_names = ImageNameMap(images);

  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);

  for (const auto& [image_id1, image_id2] : graph.image_pairs) {
    const auto image_name1 = image_names.find(image_id1);
    const auto image_name2 = image_names.find(image_id2);
    THROW_CHECK(image_name1 != image_names.end());
    THROW_CHECK(image_name2 != image_names.end());
    file << image_name1->second << " " << image_name2->second << "\n";
  }
}

void WriteMipMapDebugCsv(const std::filesystem::path& path,
                         const MipMapCandidateGraph& graph,
                         const std::vector<MipMapImageData>& images) {
  const NodeHashMap<image_t, std::string> image_names = ImageNameMap(images);

  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);

  file << "query_image_id,query_image_name,rank,candidate_image_id,"
          "candidate_image_name,score\n";

  for (const auto& row : graph.candidate_rows) {
    const auto query_name = image_names.find(row.image_id);
    THROW_CHECK(query_name != image_names.end());
    for (size_t rank = 0; rank < row.candidates.size(); ++rank) {
      const auto& candidate = row.candidates[rank];
      const auto candidate_name = image_names.find(candidate.image_id);
      THROW_CHECK(candidate_name != image_names.end());
      file << row.image_id << ",";
      WriteCsvCell(file, query_name->second);
      file << "," << rank << "," << candidate.image_id << ",";
      WriteCsvCell(file, candidate_name->second);
      file << "," << candidate.score << "\n";
    }
  }
}

}  // namespace colmap
