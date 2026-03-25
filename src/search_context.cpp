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
#include "omega/profiling_timer.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace omega {

namespace {

struct WeightedBhCacheKey {
  int k;
  int k_train;

  bool operator==(const WeightedBhCacheKey& other) const {
    return k == other.k && k_train == other.k_train;
  }
};

struct WeightedBhCacheKeyHash {
  size_t operator()(const WeightedBhCacheKey& key) const {
    return (static_cast<size_t>(static_cast<uint32_t>(key.k)) << 32) ^
           static_cast<size_t>(static_cast<uint32_t>(key.k_train));
  }
};

const std::vector<float>& GetWeightedBhRatios(int k, int k_train) {
  static std::mutex cache_mutex;
  static std::unordered_map<WeightedBhCacheKey, std::vector<float>,
                            WeightedBhCacheKeyHash>
      ratios_cache;

  const WeightedBhCacheKey key{k, k_train};
  std::lock_guard<std::mutex> lock(cache_mutex);
  auto it = ratios_cache.find(key);
  if (it != ratios_cache.end()) {
    return it->second;
  }

  auto [inserted_it, _] = ratios_cache.emplace(key, std::vector<float>(k, 0.0f));
  auto& ratios = inserted_it->second;
  if (k <= 0 || k_train <= 0) {
    return ratios;
  }

  const int block_count = (k + k_train - 1) / k_train;
  std::vector<float> weights(block_count, 0.0f);
  std::vector<float> prefix(block_count + 1, 0.0f);
  for (int block = 0; block < block_count; ++block) {
    weights[block] = 1.0f / std::sqrt(static_cast<float>(block + 1));
    prefix[block + 1] = prefix[block] + weights[block];
  }

  const float total_weight = prefix[block_count];
  if (total_weight <= 0.0f) {
    return ratios;
  }

  for (int i = 0; i < k; ++i) {
    const int included_blocks = i / k_train + 1;
    ratios[i] = prefix[included_blocks] / total_weight;
  }
  return ratios;
}

}  // namespace

SearchContext::SearchContext(const GBDTModel* model, const ModelTables* tables,
                             float target_recall, int k, int window_size)
    : model_(model),
      tables_(tables),
      target_recall_(target_recall),
      k_(k),
      window_size_(window_size),
      k_train_(1),  // Phase 4: K value for training
      use_weighted_bh_(true),  // Phase 4: Enable Weighted BH by default
      top_candidates_sorted_cache_valid_(true),
      traversal_window_head_(0),
      traversal_window_size_(0),
      hops_(0),
      comparisons_(0),
      dist_start_(std::numeric_limits<float>::max()),
      dist_1st_(std::numeric_limits<float>::max()),
      collected_gt_(0),
      next_prediction_cmps_(50),
      last_predicted_recall_avg_(0.0f),
      last_predicted_recall_at_target_(0.0f),
      early_stop_hit_(false),
      training_mode_enabled_(false),  // Phase 5: Training mode disabled by default
      current_query_id_(-1),
      should_stop_calls_(0),
      prediction_calls_(0),
      should_stop_time_ns_(0),
      prediction_eval_time_ns_(0),
      sorted_window_time_ns_(0),
      average_recall_eval_time_ns_(0),
      prediction_feature_prep_time_ns_(0),
      collected_gt_advance_count_(0),
      should_stop_calls_with_advance_(0),
      max_prediction_calls_per_should_stop_(0),
      collect_timing_(IsControlTimingEnabled()) {  // Phase 5: No query ID initially
  traversal_window_buffer_.resize(window_size_ > 0 ? window_size_ : 0);
  top_candidates_.reserve(k_ > 0 ? k_ : 0);
  sorted_window_scratch_.reserve(window_size_ > 0 ? window_size_ : 0);
  filtered_distances_scratch_.reserve(window_size_ > 0 ? window_size_ : 0);
  masked_ids_scratch_.reserve(k_ > 0 ? k_ : 0);

  // Get initial prediction interval from interval_table
  auto interval = GetPredictionInterval(target_recall);
  next_prediction_cmps_ = interval.first;  // initial_interval

  // Phase 4: Initialize Weighted BH method
  InitializeWeightedBH();
}

SearchContext::~SearchContext() = default;

void SearchContext::Reset() {
  traversal_window_head_ = 0;
  traversal_window_size_ = 0;
  top_candidates_.clear();
  top_candidates_sorted_cache_.clear();
  top_candidates_sorted_cache_valid_ = true;
  hops_ = 0;
  comparisons_ = 0;
  dist_start_ = std::numeric_limits<float>::max();
  dist_1st_ = std::numeric_limits<float>::max();
  collected_gt_ = 0;
  traversal_window_stats_cache_.clear();  // Clear stats cache
  last_predicted_recall_avg_ = 0.0f;
  last_predicted_recall_at_target_ = 0.0f;
  early_stop_hit_ = false;
  should_stop_calls_ = 0;
  prediction_calls_ = 0;
  should_stop_time_ns_ = 0;
  prediction_eval_time_ns_ = 0;
  sorted_window_time_ns_ = 0;
  average_recall_eval_time_ns_ = 0;
  prediction_feature_prep_time_ns_ = 0;
  collected_gt_advance_count_ = 0;
  should_stop_calls_with_advance_ = 0;
  max_prediction_calls_per_should_stop_ = 0;

  // Reset prediction interval
  auto interval = GetPredictionInterval(target_recall_);
  next_prediction_cmps_ = interval.first;

  // Phase 4: Re-initialize Weighted BH for new query
  InitializeWeightedBH();
}

bool SearchContext::UpdateTopCandidates(int node_id, float distance, int cmps) {
  if (k_ <= 0) {
    return false;
  }

  TopCandidate candidate{node_id, distance, cmps};
  if (TopCandidateCount() == k_) {
    const auto& worst = top_candidates_.front();
    if (!TopCandidateLess(candidate, worst)) {
      return false;
    }

    std::pop_heap(top_candidates_.begin(), top_candidates_.end(),
                  TopCandidateLess);
    top_candidates_.back() = candidate;
    std::push_heap(top_candidates_.begin(), top_candidates_.end(),
                   TopCandidateLess);
  } else {
    top_candidates_.push_back(candidate);
    std::push_heap(top_candidates_.begin(), top_candidates_.end(),
                   TopCandidateLess);
  }

  top_candidates_sorted_cache_valid_ = false;
  dist_1st_ = std::min(dist_1st_, distance);
  return true;
}

bool SearchContext::TopCandidateLess(const TopCandidate& lhs,
                                     const TopCandidate& rhs) {
  if (lhs.distance != rhs.distance) {
    return lhs.distance < rhs.distance;
  }
  return lhs.id < rhs.id;
}

const std::vector<SearchContext::TopCandidate>&
SearchContext::GetSortedTopCandidates() const {
  if (!top_candidates_sorted_cache_valid_) {
    top_candidates_sorted_cache_ = top_candidates_;
    std::sort(top_candidates_sorted_cache_.begin(),
              top_candidates_sorted_cache_.end(), TopCandidateLess);
    top_candidates_sorted_cache_valid_ = true;
  }
  return top_candidates_sorted_cache_;
}

int SearchContext::TopCandidateCount() const {
  return static_cast<int>(top_candidates_.size());
}

void SearchContext::PushTraversalWindow(int node_id, float distance) {
  if (window_size_ <= 0 || traversal_window_buffer_.empty()) {
    return;
  }

  if (traversal_window_size_ < window_size_) {
    int insert_idx = (traversal_window_head_ + traversal_window_size_) % window_size_;
    traversal_window_buffer_[insert_idx] = {node_id, distance};
    ++traversal_window_size_;
    return;
  }

  traversal_window_buffer_[traversal_window_head_] = {node_id, distance};
  traversal_window_head_ = (traversal_window_head_ + 1) % window_size_;
}

void SearchContext::CopyTraversalWindowTo(
    std::vector<std::pair<int, float>>* out) const {
  out->clear();
  if (traversal_window_size_ <= 0 || traversal_window_buffer_.empty()) {
    return;
  }

  out->reserve(traversal_window_size_);
  for (int i = 0; i < traversal_window_size_; ++i) {
    int idx = (traversal_window_head_ + i) % window_size_;
    out->push_back(traversal_window_buffer_[idx]);
  }
}

void SearchContext::ReportVisit(int node_id, float distance, bool is_in_topk) {
  comparisons_++;

  PushTraversalWindow(node_id, distance);

  // Update topk tracking
  if (is_in_topk) {
    UpdateTopCandidates(node_id, distance, comparisons_);

    // Track gt_cmps for each GT rank when the GT node first enters topk
    if (training_mode_enabled_ && !ground_truth_.empty()) {
      // Check if this node is a GT node and hasn't been found yet
      if (gt_found_set_.find(node_id) == gt_found_set_.end()) {
        // Check which GT rank(s) this node corresponds to
        for (size_t rank = 0; rank < ground_truth_.size(); ++rank) {
          if (ground_truth_[rank] == node_id) {
            // Record the cmps when this GT rank was found
            gt_cmps_per_rank_[rank] = comparisons_;
            gt_found_set_.insert(node_id);
            break;  // Each node_id can only match one GT rank
          }
        }
      }
    }
  }

  // Phase 5: Collect training features if training mode is enabled
  // MEMORY OPTIMIZED: Compute label in real-time instead of storing collected_node_ids
  // This reduces memory from O(records * ef) to O(records * 11_features)
  if (training_mode_enabled_) {
    TrainingRecord record;
    record.query_id = current_query_id_;
    record.hops = hops_;
    record.cmps = comparisons_;
    record.dist_1st = dist_1st_;
    record.dist_start = dist_start_;

    // Use cached stats (computed in ReportHop) - O(1) copy
    // If cache is empty (before first hop), use default zeros
    if (traversal_window_stats_cache_.size() == 7) {
      record.traversal_window_stats = traversal_window_stats_cache_;
    } else {
      record.traversal_window_stats = std::vector<float>(7, 0.0f);
    }

    // Match the reference generate_training_data.py semantics:
    // label becomes 1 once the first k_train ground-truth items have all been
    // collected at least once by the current comparison count. This is
    // monotonic in cmps and does not require the GT items to remain in top-k.
    record.label = 0;
    if (!ground_truth_.empty()) {
      size_t actual_k = std::min(static_cast<size_t>(k_train_), ground_truth_.size());
      bool all_found = true;
      for (size_t i = 0; i < actual_k && all_found; ++i) {
        if (i >= gt_cmps_per_rank_.size() ||
            gt_cmps_per_rank_[i] < 0 ||
            comparisons_ < gt_cmps_per_rank_[i]) {
          all_found = false;
        }
      }
      record.label = all_found ? 1 : 0;
    }

    training_records_.push_back(record);
  }
}

bool SearchContext::ReportVisitCandidate(int node_id, float distance,
                                         bool inserted_to_topk) {
  comparisons_++;
  PushTraversalWindow(node_id, distance);

  if (inserted_to_topk) {
    UpdateTopCandidates(node_id, distance, comparisons_);

    if (training_mode_enabled_ && !ground_truth_.empty() &&
        gt_found_set_.find(node_id) == gt_found_set_.end()) {
      for (size_t rank = 0; rank < ground_truth_.size(); ++rank) {
        if (ground_truth_[rank] == node_id) {
          gt_cmps_per_rank_[rank] = comparisons_;
          gt_found_set_.insert(node_id);
          break;
        }
      }
    }
  }

  if (training_mode_enabled_) {
    TrainingRecord record;
    record.query_id = current_query_id_;
    record.hops = hops_;
    record.cmps = comparisons_;
    record.dist_1st = dist_1st_;
    record.dist_start = dist_start_;

    if (traversal_window_stats_cache_.size() == 7) {
      record.traversal_window_stats = traversal_window_stats_cache_;
    } else {
      record.traversal_window_stats = std::vector<float>(7, 0.0f);
    }

    record.label = 0;
    if (!ground_truth_.empty()) {
      size_t actual_k =
          std::min(static_cast<size_t>(k_train_), ground_truth_.size());
      bool all_found = true;
      for (size_t i = 0; i < actual_k && all_found; ++i) {
        if (i >= gt_cmps_per_rank_.size() || gt_cmps_per_rank_[i] < 0 ||
            comparisons_ < gt_cmps_per_rank_[i]) {
          all_found = false;
        }
      }
      record.label = all_found ? 1 : 0;
    }

    training_records_.push_back(record);
  }
  return inserted_to_topk;
}

void SearchContext::ReportHop() {
  hops_++;

  // Phase 5: Compute traversal window stats cache for training mode
  // This is called ONCE per hop, before processing all neighbors
  // The cached stats are then reused in ReportVisit() for each neighbor
  // Matches original OMEGA design: O(window_size * log(window_size)) per hop
  // instead of O(window_size * log(window_size)) per visit
  if (training_mode_enabled_) {
    std::vector<int> masked_ids;  // Empty for training mode (like original OMEGA)
    traversal_window_stats_cache_ = GetTraversalWindowStats(masked_ids);
  }
}

bool SearchContext::ShouldPredict() const {
  // Check if we have enough results and reached prediction point
  return comparisons_ >= next_prediction_cmps_ && TopCandidateCount() >= k_;
}

void SearchContext::GetStats(int* hops, int* comparisons, int* collected_gt) const {
  if (hops) *hops = hops_;
  if (comparisons) *comparisons = comparisons_;
  if (collected_gt) *collected_gt = collected_gt_;
}

bool SearchContext::ShouldStopEarly() {
  ++should_stop_calls_;
  ProfilingTimer::tick_t should_stop_start = 0;
  if (collect_timing_) {
    should_stop_start = ProfilingTimer::Now();
  }
  uint64_t predictions_before = prediction_calls_;
  bool collected_gt_advanced_in_call = false;
  float predicted_recall_at_target = 0.0f;

  auto evaluate_average_recall = [this]() -> float {
    double predicted_recall_avg = 0.0;
    for (int i = 1; i <= k_; i++) {
      if (i <= collected_gt_) {
        predicted_recall_avg += 1.0;
      } else {
        double recall_from_gt_collected =
            GetRecallFromGtCollectedTable(collected_gt_, i);
        double recall_from_gt_cmps =
            GetRecallFromGtCmpsAllTable(i, comparisons_);
        predicted_recall_avg +=
            std::max(recall_from_gt_collected, recall_from_gt_cmps);
      }
    }
    predicted_recall_avg /= k_;
    last_predicted_recall_avg_ = static_cast<float>(predicted_recall_avg);
    return last_predicted_recall_avg_;
  };

  if (!model_ || !tables_) {
    if (collect_timing_) {
      should_stop_time_ns_ +=
          ProfilingTimer::ElapsedNs(should_stop_start, ProfilingTimer::Now());
    }
    return false;  // No model, can't make decision
  }

  // Phase 4: collected_gt iteration with per-K prediction
  if (use_weighted_bh_ && TopCandidateCount() >= k_) {
    CopyTraversalWindowTo(&sorted_window_scratch_);
    std::sort(sorted_window_scratch_.begin(), sorted_window_scratch_.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    auto can_confirm_count = [&](int confirmed_count) -> bool {
      int idx = std::min(confirmed_count - 1, k_ - 1);
      ++prediction_calls_;
      predicted_recall_at_target =
          PredictRecallForRankWithSortedWindow(idx, sorted_window_scratch_);
      last_predicted_recall_at_target_ = predicted_recall_at_target;
      return predicted_recall_at_target >= recall_targets_[idx];
    };

    // Step 1: Find the maximum confirmable rank for this search state.
    // Use exponential expansion plus binary search rather than scanning
    // rank-by-rank from 1. This keeps easy queries from paying O(k) model
    // evaluations inside a single should_stop() call.
    bool collected_gt_has_changed = false;
    last_predicted_recall_avg_ = 0.0f;
    const int block_size = std::max(1, k_train_);
    int best_confirmed = collected_gt_;
    int first_probe = std::min(k_, collected_gt_ + block_size);
    int first_failed = -1;

    if (first_probe > collected_gt_ && can_confirm_count(first_probe)) {
      best_confirmed = first_probe;

      int step = block_size;
      while (best_confirmed < k_) {
        int next_probe = std::min(k_, best_confirmed + step);
        if (next_probe <= best_confirmed) {
          break;
        }

        if (can_confirm_count(next_probe)) {
          best_confirmed = next_probe;
          step *= 2;
        } else {
          first_failed = next_probe;
          break;
        }
      }

      if (best_confirmed < k_ && first_failed > best_confirmed) {
        int low = best_confirmed;
        int high = first_failed;
        while (high - low > block_size) {
          int mid = low + (((high - low) / (2 * block_size)) * block_size);
          if (mid <= low || mid >= high) {
            break;
          }

          if (can_confirm_count(mid)) {
            low = mid;
          } else {
            high = mid;
          }
        }
        best_confirmed = low;
      }
    } else {
      first_failed = first_probe;
    }

    if (best_confirmed > collected_gt_) {
      collected_gt_has_changed = true;
      collected_gt_advanced_in_call = true;
      collected_gt_advance_count_ += (best_confirmed - collected_gt_);
      collected_gt_ = best_confirmed;
      if (collected_gt_ >= k_) {
        early_stop_hit_ = true;
        last_predicted_recall_avg_ = 1.0f;
        if (collected_gt_advanced_in_call) {
          ++should_stop_calls_with_advance_;
        }
        max_prediction_calls_per_should_stop_ =
            std::max(max_prediction_calls_per_should_stop_,
                     prediction_calls_ - predictions_before);
        if (collect_timing_) {
          should_stop_time_ns_ +=
              ProfilingTimer::ElapsedNs(should_stop_start, ProfilingTimer::Now());
        }
        return true;  // All K results confirmed!
      }
    }

    // Step 2: Compute average recall if collected_gt changed
    if (collected_gt_has_changed) {
      float predicted_recall_avg = evaluate_average_recall();

      if (predicted_recall_avg >= target_recall_) {
        early_stop_hit_ = true;
        ++should_stop_calls_with_advance_;
        max_prediction_calls_per_should_stop_ =
            std::max(max_prediction_calls_per_should_stop_,
                     prediction_calls_ - predictions_before);
        if (collect_timing_) {
          should_stop_time_ns_ +=
              ProfilingTimer::ElapsedNs(should_stop_start, ProfilingTimer::Now());
        }
        return true;  // Early stop!
      }

      auto interval = GetPredictionInterval(target_recall_);
      int min_interval = interval.second;
      int initial_interval = interval.first;
      float recall_gap = target_recall_ - static_cast<float>(predicted_recall_avg);
      int interval_adjustment = static_cast<int>(
          min_interval +
          (initial_interval - min_interval) * std::max(0.0f, recall_gap));
      next_prediction_cmps_ = comparisons_ + interval_adjustment;
      ++should_stop_calls_with_advance_;
      max_prediction_calls_per_should_stop_ =
          std::max(max_prediction_calls_per_should_stop_,
                   prediction_calls_ - predictions_before);
      if (collect_timing_) {
        should_stop_time_ns_ +=
            ProfilingTimer::ElapsedNs(should_stop_start, ProfilingTimer::Now());
      }
      return false;
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

    max_prediction_calls_per_should_stop_ =
        std::max(max_prediction_calls_per_should_stop_,
                 prediction_calls_ - predictions_before);
    if (collect_timing_) {
      should_stop_time_ns_ +=
          ProfilingTimer::ElapsedNs(should_stop_start, ProfilingTimer::Now());
    }
    return false;
  }

  // Fallback to original prediction logic if Weighted BH is disabled
  // Extract features and predict
  std::vector<float> features = ExtractFeatures();
  if (features.empty()) {
    if (collect_timing_) {
      should_stop_time_ns_ +=
          ProfilingTimer::ElapsedNs(should_stop_start, ProfilingTimer::Now());
    }
    return false;  // Not enough data yet
  }

  ++prediction_calls_;
  float predicted_recall = PredictWithFeatures(features);
  last_predicted_recall_at_target_ = predicted_recall;
  last_predicted_recall_avg_ = predicted_recall;

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
  collected_gt_ = TopCandidateCount();

  // Stop if we've reached target recall
  if (predicted_recall >= target_recall_) {
    early_stop_hit_ = true;
    if (collect_timing_) {
      should_stop_time_ns_ +=
          ProfilingTimer::ElapsedNs(should_stop_start, ProfilingTimer::Now());
    }
    return true;
  }
  if (collect_timing_) {
    should_stop_time_ns_ +=
        ProfilingTimer::ElapsedNs(should_stop_start, ProfilingTimer::Now());
  }
  return false;
}

std::vector<float> SearchContext::ExtractFeatures() {
  // Need at least some data in traversal window
  if (traversal_window_size_ <= 0) {
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
  const auto& top_candidates = GetSortedTopCandidates();
  std::vector<int> masked_ids;
  masked_ids.reserve(top_candidates.size());
  for (const auto& candidate : top_candidates) {
    masked_ids.push_back(candidate.id);
  }

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
  CopyTraversalWindowTo(&sorted_window_scratch_);
  std::sort(sorted_window_scratch_.begin(), sorted_window_scratch_.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
  return GetTraversalWindowStatsFromSortedWindow(sorted_window_scratch_,
                                                 masked_ids);
}

std::array<float, 7> SearchContext::GetTraversalWindowStatsArrayFromSortedWindow(
    const std::vector<std::pair<int, float>>& sorted_window,
    const std::vector<int>& masked_ids) {
  auto original_len = sorted_window.size();
  size_t len = 0;
  filtered_distances_scratch_.clear();
  filtered_distances_scratch_.reserve(original_len);

  float avg = 0.0f;
  float var = 0.0f;
  float min = std::numeric_limits<float>::max();
  float max = std::numeric_limits<float>::lowest();

  for (size_t i = 0; i < original_len; i++) {
    auto it = std::lower_bound(masked_ids.begin(), masked_ids.end(),
                               sorted_window[i].first);
    if (it == masked_ids.end() || *it != sorted_window[i].first) {
      ++len;
      float dist = sorted_window[i].second;
      filtered_distances_scratch_.push_back(dist);
      avg += dist;
      var += dist * dist;
      min = std::min(min, dist);
      max = std::max(max, dist);
    }
  }

  if (len == 0) {
    return {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  }
  if (len == 1) {
    return {avg, 0.0f, avg, avg, avg, avg, avg};
  }

  size_t len25 = len / 4;
  size_t len50 = len / 2;
  size_t len75 = len * 3 / 4;

  float perc25 = filtered_distances_scratch_[len25];
  float med = filtered_distances_scratch_[len50];
  float perc75 = filtered_distances_scratch_[len75];

  avg /= static_cast<float>(len);
  var = var / static_cast<float>(len) - avg * avg;

  return {avg, var, min, max, med, perc25, perc75};
}

std::vector<float> SearchContext::GetTraversalWindowStatsFromSortedWindow(
    const std::vector<std::pair<int, float>>& sorted_window,
    const std::vector<int>& masked_ids) {
  auto stats = GetTraversalWindowStatsArrayFromSortedWindow(sorted_window,
                                                            masked_ids);
  return std::vector<float>(stats.begin(), stats.end());
}

float SearchContext::PredictWithFeatures(const std::vector<float>& features) {
  if (!model_ || features.size() != 11) {
    return 0.0f;
  }

  return PredictWithFeatureArray(
      {features[0], features[1], features[2], features[3], features[4],
       features[5], features[6], features[7], features[8], features[9],
       features[10]});
}

float SearchContext::PredictWithFeatureArray(
    const std::array<float, 11>& features) {
  if (!model_) {
    return 0.0f;
  }

  ProfilingTimer::tick_t prediction_start = 0;
  if (collect_timing_) {
    prediction_start = ProfilingTimer::Now();
  }

  std::array<double, 11> features_double{};
  for (size_t i = 0; i < features.size(); ++i) {
    features_double[i] = static_cast<double>(features[i]);
  }

  // Match the reference LightGBM flow: get raw score first, then apply sigmoid
  // exactly once before threshold-table calibration.
  double raw_score =
      model_->PredictRaw(features_double.data(),
                         static_cast<int32_t>(features_double.size()));

  // Apply sigmoid to get the raw model confidence.
  double probability = 1.0 / (1.0 + std::exp(-raw_score));

  // Map the raw confidence to calibrated recall using threshold_table.
  // The threshold table is learned with isotonic regression during training so
  // that this score behaves like the true probability that the current masked
  // top-1 decision is correct. That calibration matters because Weighted BH
  // consumes these rank-wise confidence values as if they were comparable
  // probability/confidence statements; using the uncalibrated model score would
  // bias the per-rank targets and make the final top-k control less reliable.
  if (!tables_ || tables_->threshold_table.empty()) {
    if (collect_timing_) {
      prediction_eval_time_ns_ +=
          ProfilingTimer::ElapsedNs(prediction_start, ProfilingTimer::Now());
    }
    return static_cast<float>(probability);
  }

  // threshold_table maps int(probability * 10000) to recall
  int score_key = static_cast<int>(std::round(probability * 10000));

  // Find the recall value in threshold_table
  auto it = tables_->threshold_table.upper_bound(score_key);
  if (it != tables_->threshold_table.begin()) {
    --it;
  }

  if (collect_timing_) {
    prediction_eval_time_ns_ +=
        ProfilingTimer::ElapsedNs(prediction_start, ProfilingTimer::Now());
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
// OMEGA's rank-wise prediction loop effectively turns one top-k stopping
// decision into k masked top-1 confirmation problems. A false positive at an
// early rank contaminates all later masked ranks, so the per-rank prediction
// errors accumulate. Weighted BH gives us a principled way to distribute the
// overall recall budget across ranks: earlier ranks keep stricter targets,
// later ranks can relax slightly, and we do not need an extra hand-tuned
// hyperparameter to control that schedule.
void SearchContext::InitializeWeightedBH() {
  recall_targets_.resize(k_);
  initial_intervals_.resize(k_);
  min_intervals_.resize(k_);

  if (use_weighted_bh_) {
    const auto& bh_ratios = GetWeightedBhRatios(k_, k_train_);
    for (int i = 0; i < k_; i++) {
      float curr_recall_target =
          1.0f - (1.0f - target_recall_) * bh_ratios[i];
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
  const auto& top_candidates = GetSortedTopCandidates();
  if (idx < 0 || idx >= static_cast<int>(top_candidates.size())) {
    return std::vector<float>();
  }

  std::vector<float> features(11);
  features[0] = static_cast<float>(hops_);
  features[1] = static_cast<float>(comparisons_ - idx);
  features[2] = top_candidates[idx].distance;
  features[3] = dist_start_;

  // Compute masked_ids: all candidates before idx
  std::vector<int> masked_ids;
  for (int prev_idx = 0;
       prev_idx + k_train_ - 1 < idx &&
       prev_idx < static_cast<int>(top_candidates.size());
       ++prev_idx) {
    if (comparisons_ - top_candidates[prev_idx].cmps <= window_size_) {
      masked_ids.push_back(top_candidates[prev_idx].id);
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
  if (!model_ || idx < 0 || idx >= TopCandidateCount()) {
    return 0.0f;
  }

  CopyTraversalWindowTo(&sorted_window_scratch_);
  std::sort(sorted_window_scratch_.begin(), sorted_window_scratch_.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

  return PredictRecallForRankWithSortedWindow(idx, sorted_window_scratch_);
}

float SearchContext::PredictRecallForRankWithSortedWindow(
    int idx, const std::vector<std::pair<int, float>>& sorted_window) {
  const auto& top_candidates = GetSortedTopCandidates();
  if (!model_ || idx < 0 || idx >= static_cast<int>(top_candidates.size())) {
    return 0.0f;
  }

  std::array<float, 11> features{};
  features[0] = static_cast<float>(hops_);
  features[1] = static_cast<float>(comparisons_ - idx);
  features[2] = top_candidates[idx].distance;
  features[3] = dist_start_;

  masked_ids_scratch_.clear();
  for (int prev_idx = 0;
       prev_idx + k_train_ - 1 < idx &&
       prev_idx < static_cast<int>(top_candidates.size());
       ++prev_idx) {
    if (comparisons_ - top_candidates[prev_idx].cmps <= window_size_) {
      masked_ids_scratch_.push_back(top_candidates[prev_idx].id);
    }
  }
  std::sort(masked_ids_scratch_.begin(), masked_ids_scratch_.end());

  auto window_stats =
      GetTraversalWindowStatsArrayFromSortedWindow(sorted_window,
                                                   masked_ids_scratch_);
  for (size_t i = 0; i < window_stats.size(); ++i) {
    features[4 + i] = window_stats[i];
  }
  return PredictWithFeatureArray(features);
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
  int rank_idx = rank - 1;
  if (rank_idx < 0 || rank_idx >= static_cast<int>(row.size())) {
    return 0.0f;  // Out of bounds
  }

  return row[rank_idx];
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

  auto lower = std::lower_bound(percentiles.begin(), percentiles.end(),
                                static_cast<float>(cmps));
  size_t percentile_idx = std::distance(percentiles.begin(), lower) + 1;
  if (percentile_idx > percentiles.size()) {
    percentile_idx = percentiles.size();
  }
  return static_cast<float>(percentile_idx) / 100.0f;
}

// Phase 5: Enable training mode with ground truth for real-time label computation
void SearchContext::EnableTrainingMode(int query_id, const std::vector<int>& ground_truth, int k_train) {
  training_mode_enabled_ = true;
  current_query_id_ = query_id;
  ground_truth_ = ground_truth;
  k_train_ = k_train;
  traversal_window_stats_cache_.clear();  // Clear cache for new query
  top_candidates_.clear();
  top_candidates_sorted_cache_.clear();
  top_candidates_sorted_cache_valid_ = true;
  traversal_window_head_ = 0;
  traversal_window_size_ = 0;
  dist_1st_ = std::numeric_limits<float>::max();

  // Initialize gt_cmps_per_rank with -1 (will be set to total_cmps at end if not found)
  gt_cmps_per_rank_.assign(ground_truth.size(), -1);
  gt_found_set_.clear();
}

// Phase 5: Disable training mode
void SearchContext::DisableTrainingMode() {
  training_mode_enabled_ = false;
  current_query_id_ = -1;
  ground_truth_.clear();
  gt_cmps_per_rank_.clear();
  gt_found_set_.clear();
}

} // namespace omega
