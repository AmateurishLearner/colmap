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

#include "colmap/util/testing.h"

#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace {

FeatureDescriptors MakeSiftDescriptors(
    const std::vector<std::pair<uint8_t, uint8_t>>& prefixes) {
  FeatureDescriptors descriptors;
  descriptors.type = FeatureExtractorType::SIFT;
  descriptors.data.resize(prefixes.size(), 128);
  descriptors.data.setZero();
  for (Eigen::Index i = 0; i < descriptors.data.rows(); ++i) {
    descriptors.data(i, 0) = prefixes[i].first;
    descriptors.data(i, 1) = prefixes[i].second;
  }
  return descriptors;
}

std::string ReadTextFile(const std::filesystem::path& path) {
  std::ifstream file(path);
  std::stringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

}  // namespace

TEST(MipMapMatchingOptions, Check) {
  MipMapMatchingOptions options;
  EXPECT_TRUE(options.Check());

  options.num_images = 0;
  EXPECT_FALSE(options.Check());
  options.num_images = 60;

  options.num_nearest_neighbors = 0;
  EXPECT_FALSE(options.Check());
  options.num_nearest_neighbors = 5;

  options.max_num_features = 0;
  EXPECT_FALSE(options.Check());
}

TEST(CreateMipMapCandidateGraph, ExportsDeduplicatedPairs) {
  std::vector<MipMapImageData> images;
  images.push_back(
      {1, "image1.jpg", MakeSiftDescriptors({{0, 0}, {0, 1}, {0, 2}})});
  images.push_back(
      {2, "image2.jpg", MakeSiftDescriptors({{0, 0}, {0, 1}, {0, 3}})});

  MipMapMatchingOptions options;
  options.num_images = 2;
  options.num_nearest_neighbors = 4;
  options.num_threads = 1;

  const MipMapCandidateGraph graph =
      CreateMipMapCandidateGraph(options, images);

  ASSERT_EQ(graph.candidate_rows.size(), 2);
  ASSERT_EQ(graph.candidate_rows[0].candidates.size(), 2);
  ASSERT_EQ(graph.candidate_rows[1].candidates.size(), 2);
  EXPECT_EQ(graph.candidate_rows[0].candidates[0].image_id, 1);
  EXPECT_EQ(graph.candidate_rows[0].candidates[1].image_id, 2);
  EXPECT_EQ(graph.candidate_rows[1].candidates[0].image_id, 2);
  EXPECT_EQ(graph.candidate_rows[1].candidates[1].image_id, 1);

  ASSERT_EQ(graph.image_pairs.size(), 1);
  EXPECT_EQ(graph.image_pairs[0].first, 1);
  EXPECT_EQ(graph.image_pairs[0].second, 2);
}

TEST(CreateMipMapCandidateGraph, FillsMissingCandidatesWithZeroScore) {
  std::vector<MipMapImageData> images;
  images.push_back({1, "image1.jpg", MakeSiftDescriptors({{0, 0}})});
  images.push_back({2, "image2.jpg", MakeSiftDescriptors({{100, 0}})});
  images.push_back({3, "image3.jpg", MakeSiftDescriptors({{200, 0}})});

  MipMapMatchingOptions options;
  options.num_images = 3;
  options.num_nearest_neighbors = 1;
  options.num_threads = 1;

  const MipMapCandidateGraph graph =
      CreateMipMapCandidateGraph(options, images);

  ASSERT_EQ(graph.candidate_rows.size(), 3);
  for (const auto& row : graph.candidate_rows) {
    ASSERT_EQ(row.candidates.size(), 3);
    EXPECT_EQ(row.candidates[0].image_id, row.image_id);
  }
}

TEST(WriteMipMapPairsText, WritesImportedPairsFormat) {
  std::vector<MipMapImageData> images;
  images.push_back({1, "image1.jpg", MakeSiftDescriptors({{0, 0}})});
  images.push_back({2, "folder/image2.jpg", MakeSiftDescriptors({{0, 1}})});

  MipMapCandidateGraph graph;
  graph.image_pairs.emplace_back(1, 2);

  const auto test_dir = CreateTestDir();
  const auto pairs_path = test_dir / "pairs.txt";
  WriteMipMapPairsText(pairs_path, graph, images);

  EXPECT_EQ(ReadTextFile(pairs_path), "image1.jpg folder/image2.jpg\n");
}

TEST(WriteMipMapDebugCsv, QuotesImageNames) {
  std::vector<MipMapImageData> images;
  images.push_back({1, "image,1.jpg", MakeSiftDescriptors({{0, 0}})});
  images.push_back({2, "image\"2.jpg", MakeSiftDescriptors({{0, 1}})});

  MipMapCandidateGraph graph;
  MipMapImageCandidateRow row;
  row.image_id = 1;
  row.candidates.push_back({1, std::numeric_limits<double>::infinity()});
  row.candidates.push_back({2, 0.5});
  graph.candidate_rows.push_back(std::move(row));

  const auto test_dir = CreateTestDir();
  const auto debug_path = test_dir / "debug.csv";
  WriteMipMapDebugCsv(debug_path, graph, images);

  const std::string debug_text = ReadTextFile(debug_path);
  EXPECT_NE(debug_text.find("\"image,1.jpg\""), std::string::npos);
  EXPECT_NE(debug_text.find("\"image\"\"2.jpg\""), std::string::npos);
}

}  // namespace colmap
