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

#include "omega/feature_extractor.h"
#include <cmath>
#include <algorithm>

namespace omega {

FeatureExtractor::FeatureExtractor(size_t window_size)
    : window_size_(window_size) {}

std::vector<float> FeatureExtractor::Extract(const SearchState& state) const {
  std::vector<float> features;
  features.reserve(11);

  // Basic features
  features.push_back(static_cast<float>(state.curr_hops));
  features.push_back(static_cast<float>(state.curr_cmps));
  features.push_back(state.dist_1st);
  features.push_back(state.dist_start);

  // Window statistics (7 features)
  std::vector<float> window_stats = ExtractWindowStats(state.distance_window);
  features.insert(features.end(), window_stats.begin(), window_stats.end());

  return features;
}

void FeatureExtractor::ComputeStats(const std::vector<float>& distances,
                                   float* mean, float* variance) const {
  if (distances.empty()) {
    *mean = 0.0f;
    *variance = 0.0f;
    return;
  }

  // Compute mean
  float sum = 0.0f;
  for (float dist : distances) {
    sum += dist;
  }
  *mean = sum / distances.size();

  // Compute variance
  float var_sum = 0.0f;
  for (float dist : distances) {
    float diff = dist - *mean;
    var_sum += diff * diff;
  }
  *variance = var_sum / distances.size();
}

std::vector<float> FeatureExtractor::ExtractWindowStats(
    const std::vector<float>& distance_window) const {
  std::vector<float> stats;
  stats.reserve(7);

  if (distance_window.empty()) {
    // Return zeros if no distances available
    stats.resize(7, 0.0f);
    return stats;
  }

  // Compute statistics for different window segments
  // This follows the OMEGA paper's approach of using multiple window statistics

  // Full window mean and variance
  float full_mean, full_var;
  ComputeStats(distance_window, &full_mean, &full_var);
  stats.push_back(full_mean);
  stats.push_back(full_var);

  // Recent half window mean and variance
  size_t half_size = distance_window.size() / 2;
  if (half_size > 0) {
    std::vector<float> recent_half(
        distance_window.end() - half_size, distance_window.end());
    float recent_mean, recent_var;
    ComputeStats(recent_half, &recent_mean, &recent_var);
    stats.push_back(recent_mean);
    stats.push_back(recent_var);
  } else {
    stats.push_back(full_mean);
    stats.push_back(full_var);
  }

  // Recent quarter window mean and variance
  size_t quarter_size = distance_window.size() / 4;
  if (quarter_size > 0) {
    std::vector<float> recent_quarter(
        distance_window.end() - quarter_size, distance_window.end());
    float quarter_mean, quarter_var;
    ComputeStats(recent_quarter, &quarter_mean, &quarter_var);
    stats.push_back(quarter_mean);
    stats.push_back(quarter_var);
  } else {
    stats.push_back(full_mean);
    stats.push_back(full_var);
  }

  // Minimum distance in window
  float min_dist = *std::min_element(distance_window.begin(),
                                     distance_window.end());
  stats.push_back(min_dist);

  return stats;
}

} // namespace omega
