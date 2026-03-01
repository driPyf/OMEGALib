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
      k_train_(1),  // Phase 4: K value for training
      use_weighted_bh_(true),  // Phase 4: Enable Weighted BH by default
      hops_(0),
      comparisons_(0),
      dist_start_(std::numeric_limits<float>::max()),
      dist_1st_(std::numeric_limits<float>::max()),
      collected_gt_(0),
      next_prediction_cmps_(50),
      training_mode_enabled_(false),  // Phase 5: Training mode disabled by default
      current_query_id_(-1) {  // Phase 5: No query ID initially
  // Get initial prediction interval from interval_table
  auto interval = GetPredictionInterval(target_recall);
  next_prediction_cmps_ = interval.first;  // initial_interval

  // Phase 4: Initialize Weighted BH method
  InitializeWeightedBH();
}

SearchContext::~SearchContext() = default;

void SearchContext::Reset() {
  traversal_window_.clear();
  topk_node_ids_.clear();
  topk_node_ids_ordered_.clear();  // Phase 4: Clear ordered candidate list
  hops_ = 0;
  comparisons_ = 0;
  dist_start_ = std::numeric_limits<float>::max();
  dist_1st_ = std::numeric_limits<float>::max();
  collected_gt_ = 0;

  // Reset prediction interval
  auto interval = GetPredictionInterval(target_recall_);
  next_prediction_cmps_ = interval.first;

  // Phase 4: Re-initialize Weighted BH for new query
  InitializeWeightedBH();
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
    topk_node_ids_ordered_.push_back(node_id);  // Phase 4: Track insertion order

    // Update dist_1st (best distance in topk)
    if (distance < dist_1st_) {
      dist_1st_ = distance;
    }
  }

  // Phase 5: Collect training features if training mode is enabled
  if (training_mode_enabled_) {
    TrainingRecord record;
    record.query_id = current_query_id_;
    record.hops = hops_;
    record.cmps = comparisons_;
    record.dist_1st = dist_1st_;
    record.dist_start = dist_start_;

    // Extract traversal window stats (7 dimensions)
    std::vector<int> masked_ids(topk_node_ids_.begin(), topk_node_ids_.end());
    std::sort(masked_ids.begin(), masked_ids.end());
    record.traversal_window_stats = GetTraversalWindowStats(masked_ids);

    // Record current topk node IDs (copy from set to vector)
    record.collected_node_ids.assign(topk_node_ids_.begin(), topk_node_ids_.end());

    record.label = 0;  // Will be filled after search completes

    training_records_.push_back(record);
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

  // Phase 4: collected_gt iteration with per-K prediction
  if (use_weighted_bh_ && static_cast<int>(topk_node_ids_ordered_.size()) >= k_) {
    // Step 1: Update collected_gt using per-K prediction
    bool collected_gt_has_changed = false;
    float predicted_recall_at_target = 0.0f;

    while (true) {
      int idx = std::min(collected_gt_ + k_train_ - 1, k_ - 1);
      predicted_recall_at_target = PredictRecallForRank(idx);

      if (predicted_recall_at_target >= recall_targets_[idx]) {
        collected_gt_has_changed = true;
        collected_gt_ += k_train_;
        if (collected_gt_ >= k_) {
          return true;  // All K results confirmed!
        }
      } else {
        break;
      }
    }

    // Step 2: Compute average recall if collected_gt changed
    if (collected_gt_has_changed) {
      double predicted_recall_avg = 0.0;
      for (int i = 1; i <= k_; i++) {
        if (i <= collected_gt_) {
          predicted_recall_avg += 1.0;
        } else {
          double recall_from_gt_collected = GetRecallFromGtCollectedTable(collected_gt_, i);
          double recall_from_gt_cmps = GetRecallFromGtCmpsAllTable(i, comparisons_);
          predicted_recall_avg += std::max(recall_from_gt_collected, recall_from_gt_cmps);
        }
      }
      predicted_recall_avg /= k_;

      if (predicted_recall_avg >= target_recall_) {
        return true;  // Early stop!
      }
    }

    // Update next prediction interval for next iteration
    if (collected_gt_ < k_) {
      int idx = std::min(collected_gt_ + k_train_ - 1, k_ - 1);
      float recall_gap = recall_targets_[idx] - predicted_recall_at_target;
      int interval_adjustment = static_cast<int>(
          min_intervals_[idx] +
          (initial_intervals_[idx] - min_intervals_[idx]) * std::max(0.0f, recall_gap));
      next_prediction_cmps_ = comparisons_ + interval_adjustment;
    }

    return false;
  }

  // Fallback to original prediction logic if Weighted BH is disabled
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

// Phase 4: Initialize Weighted BH method
void SearchContext::InitializeWeightedBH() {
  recall_targets_.resize(k_);
  initial_intervals_.resize(k_);
  min_intervals_.resize(k_);

  if (use_weighted_bh_) {
    for (int i = 0; i < k_; i++) {
      float a = 0.0f, b = 0.0f;
      for (int j = 1; j <= i + 1; j += k_train_) {
        a += 1.0f / std::sqrt((j - 1) / k_train_ + 1);
      }
      for (int j = 1; j <= k_; j += k_train_) {
        b += 1.0f / std::sqrt((j - 1) / k_train_ + 1);
      }
      float curr_recall_target = 1.0f - (1.0f - target_recall_) * (a / b);
      float mid_recall_target = (1.0f + target_recall_) / 2.0f;
      recall_targets_[i] = std::min(mid_recall_target, curr_recall_target);

      auto interval = GetPredictionInterval(curr_recall_target);
      initial_intervals_[i] = interval.first;
      min_intervals_[i] = interval.second;
    }
  }
}

// Phase 4: Extract features for specific rank (per-K prediction)
std::vector<float> SearchContext::ExtractFeaturesForRank(int idx) {
  std::vector<float> features(11);
  features[0] = static_cast<float>(hops_);
  features[1] = static_cast<float>(comparisons_ - idx);  // Key: cmps - idx
  features[2] = dist_1st_;
  features[3] = dist_start_;

  // Compute masked_ids: all candidates before idx
  std::vector<int> masked_ids;
  for (int i = 0; i < idx && i < static_cast<int>(topk_node_ids_ordered_.size()); i++) {
    if (comparisons_ - i <= window_size_) {
      masked_ids.push_back(topk_node_ids_ordered_[i]);
    }
  }
  std::sort(masked_ids.begin(), masked_ids.end());

  std::vector<float> window_stats = GetTraversalWindowStats(masked_ids);
  for (size_t i = 0; i < 7; ++i) {
    features[4 + i] = window_stats[i];
  }
  return features;
}

// Phase 4: Predict recall for specific rank
float SearchContext::PredictRecallForRank(int idx) {
  if (!model_ || idx >= k_) {
    return 0.0f;
  }

  std::vector<float> features = ExtractFeaturesForRank(idx);
  return PredictWithFeatures(features);
}

// Phase 4: Query gt_collected_table
float SearchContext::GetRecallFromGtCollectedTable(int collected, int rank) {
  if (!tables_ || tables_->gt_collected_table.empty()) {
    return 0.0f;
  }

  // 2D lookup: gt_collected_table[collected][rank]
  auto it = tables_->gt_collected_table.find(collected);
  if (it == tables_->gt_collected_table.end()) {
    // If exact row not found, find closest row
    auto closest_it = tables_->gt_collected_table.upper_bound(collected);
    if (closest_it != tables_->gt_collected_table.begin()) {
      --closest_it;
      it = closest_it;
    } else {
      return 0.0f;
    }
  }

  const std::vector<float>& row = it->second;
  if (rank < 0 || rank >= static_cast<int>(row.size())) {
    return 0.0f;  // Out of bounds
  }

  return row[rank];
}

// Phase 4: Query gt_cmps_all_table
float SearchContext::GetRecallFromGtCmpsAllTable(int rank, int cmps) {
  if (!tables_ || tables_->gt_cmps_all_table.empty()) {
    return 0.0f;
  }

  // 2D lookup: gt_cmps_all_table[rank][percentile_idx]
  // The table contains 100 percentiles for each rank
  auto it = tables_->gt_cmps_all_table.find(rank);
  if (it == tables_->gt_cmps_all_table.end()) {
    return 0.0f;
  }

  const std::vector<float>& percentiles = it->second;
  if (percentiles.empty()) {
    return 0.0f;
  }

  // Binary search to find the percentile that matches the cmps value
  // percentiles array contains cmps values at percentiles 1%, 2%, ..., 100%
  // We want to find the highest percentile where cmps_value <= our cmps

  // Find the first percentile where cmps_value > our cmps
  auto upper = std::upper_bound(percentiles.begin(), percentiles.end(), static_cast<float>(cmps));

  if (upper == percentiles.begin()) {
    // cmps is less than the 1st percentile
    return 0.0f;
  }

  // Calculate the percentile (1-based)
  size_t percentile_idx = std::distance(percentiles.begin(), upper);

  // Return the percentile as a fraction (0.0 to 1.0)
  return static_cast<float>(percentile_idx) / 100.0f;
}

// Phase 5: Enable training mode
void SearchContext::EnableTrainingMode(int query_id) {
  training_mode_enabled_ = true;
  current_query_id_ = query_id;
}

// Phase 5: Disable training mode
void SearchContext::DisableTrainingMode() {
  training_mode_enabled_ = false;
  current_query_id_ = -1;
}

} // namespace omega
