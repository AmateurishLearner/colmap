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

#pragma once

#include "colmap/feature/types.h"
#include "colmap/scene/database.h"
#include "colmap/util/types.h"

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace colmap {

struct MipMapMatchingOptions {
  // Number of candidate images per query image. This includes the query image
  // itself, matching the recovered MipMap top-60 selector behavior.
  int num_images = 60;

  // Number of global descriptor hits used for each query descriptor.
  int num_nearest_neighbors = 5;

  // Number of threads used by the descriptor search backend.
  int num_threads = -1;

  // Optional per-image descriptor cap for quick experiments. A value <= 0 uses
  // all descriptors.
  int max_num_features = -1;

  bool Check() const;
};

struct MipMapImageData {
  image_t image_id = kInvalidImageId;
  std::string image_name;
  FeatureDescriptors descriptors;
};

struct MipMapImageCandidate {
  image_t image_id = kInvalidImageId;
  double score = 0.0;
};

struct MipMapImageCandidateRow {
  image_t image_id = kInvalidImageId;
  std::vector<MipMapImageCandidate> candidates;
};

struct MipMapCandidateGraph {
  std::vector<MipMapImageCandidateRow> candidate_rows;
  std::vector<std::pair<image_t, image_t>> image_pairs;
};

std::vector<MipMapImageData> ReadMipMapImageDataFromDatabase(
    const Database& database, int max_num_features = -1);

MipMapCandidateGraph CreateMipMapCandidateGraph(
    const MipMapMatchingOptions& options,
    const std::vector<MipMapImageData>& images);

void WriteMipMapPairsText(const std::filesystem::path& path,
                          const MipMapCandidateGraph& graph,
                          const std::vector<MipMapImageData>& images);

void WriteMipMapDebugCsv(const std::filesystem::path& path,
                         const MipMapCandidateGraph& graph,
                         const std::vector<MipMapImageData>& images);

}  // namespace colmap
