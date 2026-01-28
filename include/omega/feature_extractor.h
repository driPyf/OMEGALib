// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ZVEC_THIRDPARTY_OMEGA_INCLUDE_FEATURE_EXTRACTOR_H_
#define ZVEC_THIRDPARTY_OMEGA_INCLUDE_FEATURE_EXTRACTOR_H_

#include <vector>
#include <cstddef>

namespace omega {

// Represents the current state of a search operation.
// Used to extract features for the OMEGA decision tree model.
struct SearchState {
  // Current hop count in the HNSW graph traversal
  int curr_hops;

  // Current number of distance comparisons performed
  int curr_cmps;

  // Distance to the first (closest) result found so far
  float dist_1st;

  // Distance from query to the starting node
  float dist_start;

  // Recent distances encountered during traversal (sliding window)
  std::vector<float> distance_window;

  SearchState()
      : curr_hops(0),
        curr_cmps(0),
        dist_1st(0.0f),
        dist_start(0.0f) {}
};

// Extracts features from search state for OMEGA model prediction.
// Features include:
// - curr_hops: current hop count
// - curr_cmps: current comparison count
// - dist_1st: distance to first result
// - dist_start: distance to start node
// - 7 window statistics: mean and variance of recent distances
class FeatureExtractor {
 public:
  // Window size for computing distance statistics
  static constexpr size_t kDefaultWindowSize = 100;

  explicit FeatureExtractor(size_t window_size = kDefaultWindowSize);

  // Extract 11-dimensional feature vector from search state.
  // Returns a vector of size 11 with the following features:
  // [0]: curr_hops
  // [1]: curr_cmps
  // [2]: dist_1st
  // [3]: dist_start
  // [4-10]: 7 window statistics (mean/variance of distance windows)
  std::vector<float> Extract(const SearchState& state) const;

 private:
  size_t window_size_;

  // Compute mean and variance of a distance window
  void ComputeStats(const std::vector<float>& distances,
                   float* mean, float* variance) const;

  // Extract window statistics from the distance window
  // Returns 7 statistics values
  std::vector<float> ExtractWindowStats(
      const std::vector<float>& distance_window) const;
};

} // namespace omega

#endif  // ZVEC_THIRDPARTY_OMEGA_INCLUDE_FEATURE_EXTRACTOR_H_
