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

#include "omega/search_context.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace omega {

SearchContext::SearchContext(const GBDTModel* model, const ModelTables* tables,
                             float target_recall, int k, int window_size)
    : model_(model),
      tables_(tables),
      target_recall_(target_recall),
      k_(k),
      window_size_(window_size),
      hops_(0),
      comparisons_(0),
      dist_start_(std::numeric_limits<float>::max()),
      dist_1st_(std::numeric_limits<float>::max()),
      collected_gt_(0),
      next_prediction_cmps_(50) {
  // Get initial prediction interval from interval_table
  auto interval = GetPredictionInterval(target_recall);
  next_prediction_cmps_ = interval.first;  // initial_interval
}

SearchContext::~SearchContext() = default;

void SearchContext::Reset() {
  traversal_window_.clear();
  topk_node_ids_.clear();
  hops_ = 0;
  comparisons_ = 0;
  dist_start_ = std::numeric_limits<float>::max();
  dist_1st_ = std::numeric_limits<float>::max();
  collected_gt_ = 0;

  // Reset prediction interval
  auto interval = GetPredictionInterval(target_recall_);
  next_prediction_cmps_ = interval.first;
}

void SearchContext::ReportVisit(int node_id, float distance, bool is_in_topk) {
  comparisons_++;

  // Update traversal window
  traversal_window_.push_back({node_id, distance});
  if (traversal_window_.size() > static_cast<size_t>(window_size_)) {
    traversal_window_.pop_front();
  }

  // Update topk tracking
  if (is_in_topk) {
    topk_node_ids_.insert(node_id);
    // Update dist_1st (best distance in topk)
    if (distance < dist_1st_) {
      dist_1st_ = distance;
    }
  }
}

void SearchContext::ReportHop() {
  hops_++;
}

bool SearchContext::ShouldPredict() const {
  // Check if we have enough results and reached prediction point
  return comparisons_ >= next_prediction_cmps_ &&
         static_cast<int>(topk_node_ids_.size()) >= k_;
}

void SearchContext::GetStats(int* hops, int* comparisons, int* collected_gt) const {
  if (hops) *hops = hops_;
  if (comparisons) *comparisons = comparisons_;
  if (collected_gt) *collected_gt = collected_gt_;
}

bool SearchContext::ShouldStopEarly() {
  if (!model_ || !tables_) {
    return false;  // No model, can't make decision
  }

  // Extract features and predict
  std::vector<float> features = ExtractFeatures();
  if (features.empty()) {
    return false;  // Not enough data yet
  }

  float predicted_recall = PredictWithFeatures(features);

  // Update next prediction point based on current recall
  auto interval = GetPredictionInterval(target_recall_);
  int min_interval = interval.second;
  int initial_interval = interval.first;

  // Adjust next prediction based on how close we are to target
  float recall_gap = target_recall_ - predicted_recall;
  int interval_adjustment = static_cast<int>(
      min_interval + (initial_interval - min_interval) * std::max(0.0f, recall_gap));
  next_prediction_cmps_ = comparisons_ + interval_adjustment;

  // Update collected_gt based on topk
  collected_gt_ = static_cast<int>(topk_node_ids_.size());

  // Stop if we've reached target recall
  return predicted_recall >= target_recall_;
}

std::vector<float> SearchContext::ExtractFeatures() {
  // Need at least some data in traversal window
  if (traversal_window_.empty()) {
    return std::vector<float>();
  }

  // Extract 11-dimensional features:
  // [curr_hops, curr_cmps, dist_1st, dist_start, avg, var, min, max, med, perc25, perc75]
  std::vector<float> features(11);

  features[0] = static_cast<float>(hops_);
  features[1] = static_cast<float>(comparisons_);
  features[2] = dist_1st_;
  features[3] = dist_start_;

  // Get 7-dim traversal window statistics
  // masked_ids: nodes already in topk (for per-K prediction)
  std::vector<int> masked_ids(topk_node_ids_.begin(), topk_node_ids_.end());
  std::sort(masked_ids.begin(), masked_ids.end());

  std::vector<float> window_stats = GetTraversalWindowStats(masked_ids);

  // Copy window stats to features[4..10]
  for (size_t i = 0; i < window_stats.size() && i < 7; ++i) {
    features[4 + i] = window_stats[i];
  }

  return features;
}

std::vector<float> SearchContext::GetTraversalWindowStats(
    const std::vector<int>& masked_ids) {
  // Sort traversal_window by distance
  std::vector<std::pair<int, float>> sorted_window(
      traversal_window_.begin(), traversal_window_.end());
  std::sort(sorted_window.begin(), sorted_window.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

  auto original_len = sorted_window.size();
  size_t len = 0;
  std::vector<float> distances;
  distances.reserve(original_len);

  // stats: avg, var, min, max, med, perc25, perc75
  float avg = 0;
  float var = 0;
  float min = std::numeric_limits<float>::max();
  float max = std::numeric_limits<float>::min();

  for (size_t i = 0; i < original_len; i++) {
    // Check if this node is masked
    auto it = std::lower_bound(masked_ids.begin(), masked_ids.end(),
                               sorted_window[i].first);
    // curr id is not masked
    if (it == masked_ids.end() || *it != sorted_window[i].first) {
      len++;
      float dist = sorted_window[i].second;
      distances.push_back(dist);
      avg += dist;
      var += dist * dist;
      min = std::min(min, dist);
      max = std::max(max, dist);
    }
  }

  if (len == 0) {
    return std::vector<float>({0, 0, 0, 0, 0, 0, 0});
  }
  if (len == 1) {
    return std::vector<float>({avg, 0, avg, avg, avg, avg, avg});
  }

  size_t len25 = len / 4;
  size_t len50 = len / 2;
  size_t len75 = len * 3 / 4;

  float perc25 = distances[len25];
  float med = distances[len50];
  float perc75 = distances[len75];

  avg /= len;
  var = var / len - avg * avg;

  return std::vector<float>({avg, var, min, max, med, perc25, perc75});
}

float SearchContext::PredictWithFeatures(const std::vector<float>& features) {
  if (!model_ || features.size() != 11) {
    return 0.0f;
  }

  // Convert float features to double for model prediction
  std::vector<double> features_double(features.begin(), features.end());

  // Get model prediction (raw score)
  double raw_score = model_->Predict(features_double.data(), features_double.size());

  // Apply sigmoid to get probability
  double probability = 1.0 / (1.0 + std::exp(-raw_score));

  // Map probability to recall using threshold_table
  if (!tables_ || tables_->threshold_table.empty()) {
    return static_cast<float>(probability);
  }

  // threshold_table maps int(probability * 10000) to recall
  int score_key = static_cast<int>(std::round(probability * 10000));

  // Find the recall value in threshold_table
  auto it = tables_->threshold_table.upper_bound(score_key);
  if (it != tables_->threshold_table.begin()) {
    --it;
  }

  return it->second;
}

std::pair<int, int> SearchContext::GetPredictionInterval(float target_recall) {
  if (!tables_ || tables_->interval_table.empty()) {
    return {50, 10};  // Default: initial_interval=50, min_interval=10
  }

  // interval_table maps int(recall * 100) to (initial_interval, min_interval)
  int recall_key = static_cast<int>(std::round(target_recall * 100));

  auto it = tables_->interval_table.lower_bound(recall_key);
  if (it != tables_->interval_table.end()) {
    return it->second;
  }

  // If not found, use the largest available
  if (!tables_->interval_table.empty()) {
    return tables_->interval_table.rbegin()->second;
  }

  return {50, 10};
}

} // namespace omega
