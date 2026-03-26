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

#ifndef ZVEC_THIRDPARTY_OMEGA_INCLUDE_SEARCH_CONTEXT_H_
#define ZVEC_THIRDPARTY_OMEGA_INCLUDE_SEARCH_CONTEXT_H_

#include "omega/tree_inference.h"
#include "omega/model_manager.h"
#include "omega/feature_extractor.h"
#include <limits>
#include <array>
#include <functional>
#include <vector>
#include <unordered_set>
#include <cstdint>

namespace omega {

// Training record structure for collecting features during search
// Memory-optimized: labels are computed in real-time, no need to store collected_node_ids
struct TrainingRecord {
  int query_id;
  int hops;
  int cmps;
  float dist_1st;
  float dist_start;
  std::vector<float> traversal_window_stats;  // 7 dimensions
  int label;  // Computed in real-time during search (0 or 1)
};

// Manages the state of an adaptive search operation using OMEGA.
// Maintains search statistics, calls the GBDT model for predictions,
// and determines when to stop early based on target recall.
// This is a stateful interface - zvec reports each node visit.
class SearchContext {
 public:
  struct VisitCandidate {
    int id;
    float distance;
    bool inserted_to_topk;
  };

  struct TopCandidate {
    int id;
    float distance;
    int cmps;
  };

  // Create a search context with parameters
  SearchContext(const GBDTModel* model, const ModelTables* tables,
                float target_recall, int k, int window_size);

  ~SearchContext();

  // Reset the search state for a new query
  void Reset();

  // Report a node visit and let SearchContext maintain the result-set-sized
  // top-k structure directly.
  // Returns: true if we've reached a prediction point and have enough
  // candidates, false otherwise.
  bool ReportVisitCandidate(int node_id, float distance, bool inserted_to_topk);
  bool ReportVisitCandidates(const VisitCandidate* candidates, size_t count);

  // Report a hop during search
  void ReportHop();

  // Set the distance to the start node (called once at search start)
  void SetDistStart(float dist_start) { dist_start_ = dist_start; }

  // Check if should track traversal window
  // Only track when close to next prediction point
  bool ShouldTrackTraversalWindow() const;

  // Predict whether to stop early based on current state and target recall.
  // Returns true if the search should stop, false to continue.
  bool ShouldStopEarly();

  // Get current search statistics
  void GetStats(int* hops, int* comparisons, int* collected_gt) const;
  float GetLastPredictedRecallAvg() const { return last_predicted_recall_avg_; }
  float GetLastPredictedRecallAtTarget() const {
    return last_predicted_recall_at_target_;
  }
  bool EarlyStopHit() const { return early_stop_hit_; }
  uint64_t GetShouldStopCalls() const { return should_stop_calls_; }
  uint64_t GetPredictionCalls() const { return prediction_calls_; }
  uint64_t GetShouldStopTimeNs() const { return should_stop_time_ns_; }
  uint64_t GetPredictionEvalTimeNs() const { return prediction_eval_time_ns_; }
  uint64_t GetSortedWindowTimeNs() const { return sorted_window_time_ns_; }
  uint64_t GetAverageRecallEvalTimeNs() const {
    return average_recall_eval_time_ns_;
  }
  uint64_t GetPredictionFeaturePrepTimeNs() const {
    return prediction_feature_prep_time_ns_;
  }
  uint64_t GetReportVisitCandidateTimeNs() const {
    return report_visit_candidate_time_ns_;
  }
  uint64_t GetReportHopTimeNs() const { return report_hop_time_ns_; }
  uint64_t GetUpdateTopCandidatesTimeNs() const {
    return update_top_candidates_time_ns_;
  }
  uint64_t GetPushTraversalWindowTimeNs() const {
    return push_traversal_window_time_ns_;
  }
  uint64_t GetCollectedGtAdvanceCount() const {
    return collected_gt_advance_count_;
  }
  uint64_t GetShouldStopCallsWithAdvance() const {
    return should_stop_calls_with_advance_;
  }
  uint64_t GetMaxPredictionCallsPerShouldStop() const {
    return max_prediction_calls_per_should_stop_;
  }

  // Training mode methods (Phase 5)
  // ground_truth: top-k ground truth node IDs for this query (used to compute labels in real-time)
  // k_train: number of ground truth nodes to check (default 1)
  void EnableTrainingMode(int query_id, const std::vector<int>& ground_truth, int k_train = 1);
  const std::vector<TrainingRecord>& GetTrainingRecords() const { return training_records_; }

  // Get gt_cmps data: cmps value when each GT rank was first found in topk
  // Returns vector of size ground_truth.size(), where gt_cmps[rank] = cmps when GT[rank] was found
  // If GT[rank] was never found, returns total_cmps (the final cmps count)
  const std::vector<int>& GetGtCmpsPerRank() const { return gt_cmps_per_rank_; }
  int GetTotalCmps() const { return comparisons_; }
  int GetNextPredictionCmps() const { return next_prediction_cmps_; }
  int GetTopCandidateCountForHook() const { return TopCandidateCount(); }
  int GetK() const { return k_; }
  int GetPredictionBatchMinInterval() const;

 private:
  const GBDTModel* model_;
  const ModelTables* tables_;

  // Search parameters
  float target_recall_;
  int k_;
  int window_size_;
  int k_train_;  // K value used for training (default 1)
  // OMEGA treats top-k early stopping as k rank-wise "is the current top-1 for
  // this masked view already good enough?" decisions. Those per-rank decisions
  // are not independent: an optimistic decision at an earlier rank propagates
  // into later masked ranks and compounds the final top-k recall error.
  //
  // Weighted Benjamini-Hochberg gives us a simple way to allocate different
  // recall/confidence targets across ranks without introducing an extra tuning
  // knob. Earlier ranks receive stricter targets, later ranks receive slightly
  // looser ones, and the aggregate decision better matches the desired overall
  // top-k recall under this error accumulation pattern.
  // Reference: Benjamini and Hochberg (1995), "Controlling the False
  // Discovery Rate: A Practical and Powerful Approach to Multiple Testing",
  // JRSS B, https://doi.org/10.1111/j.2517-6161.1995.tb02031.x
  bool use_weighted_bh_;  // Use Weighted BH rank-wise recall targets.

  // Search state
  std::vector<std::pair<int, float>> traversal_window_buffer_;  // Ring buffer storage.
  std::vector<TopCandidate> top_candidates_;  // Current top-K candidates as a bounded heap.
  mutable std::vector<TopCandidate> top_candidates_sorted_cache_;
  mutable bool top_candidates_sorted_cache_valid_;
  int traversal_window_head_;
  int traversal_window_size_;
  int hops_;
  int comparisons_;
  float dist_start_;
  float dist_1st_;
  int collected_gt_;  // Number of ground truth collected
  int next_prediction_cmps_;  // When to predict next
  float last_predicted_recall_avg_;
  float last_predicted_recall_at_target_;
  bool early_stop_hit_;

  // Weighted BH state (Phase 4)
  // These arrays hold the per-rank targets/intervals derived from the global
  // target recall. They let SearchContext confirm ranks incrementally while
  // accounting for the fact that OMEGA decomposes top-k into multiple masked
  // top-1 style decisions whose errors can accumulate across ranks.
  std::vector<float> recall_targets_;  // Recall target for each rank
  std::vector<int> initial_intervals_;  // Initial prediction interval for each rank
  std::vector<int> min_intervals_;  // Minimum prediction interval for each rank

  // Training mode state (Phase 5)
  bool training_mode_enabled_;
  int current_query_id_;
  std::vector<TrainingRecord> training_records_;
  std::vector<float> traversal_window_stats_cache_;  // Computed per-hop, reused per-visit
  std::vector<int> ground_truth_;  // Ground truth node IDs for current query (for real-time label computation)
  std::vector<int> gt_cmps_per_rank_;  // cmps value when each GT rank was first found in topk
  std::unordered_set<int> gt_found_set_;  // Set of GT node IDs already found (for O(1) lookup)

  // Hot-path runtime statistics for diagnosing query-side overhead.
  uint64_t should_stop_calls_;
  uint64_t prediction_calls_;
  uint64_t should_stop_time_ns_;
  uint64_t prediction_eval_time_ns_;
  uint64_t sorted_window_time_ns_;
  uint64_t average_recall_eval_time_ns_;
  uint64_t prediction_feature_prep_time_ns_;
  uint64_t report_visit_candidate_time_ns_;
  uint64_t report_hop_time_ns_;
  uint64_t update_top_candidates_time_ns_;
  uint64_t push_traversal_window_time_ns_;
  uint64_t collected_gt_advance_count_;
  uint64_t should_stop_calls_with_advance_;
  uint64_t max_prediction_calls_per_should_stop_;
  mutable bool collect_timing_;

  // Initialize Weighted BH method (Phase 4)
  void InitializeWeightedBH();

  // Maintain the current top-k candidates like the reference implementation.
  bool UpdateTopCandidates(int node_id, float distance, int cmps);
  bool ProcessVisitCandidate(const VisitCandidate& candidate);
  static bool TopCandidateLess(const TopCandidate& lhs,
                               const TopCandidate& rhs);
  const std::vector<TopCandidate>& GetSortedTopCandidates() const;
  int TopCandidateCount() const;
  void PushTraversalWindow(int node_id, float distance);
  void CopyTraversalWindowTo(std::vector<std::pair<int, float>>* out) const;

  // Extract 11-dimensional features from current state
  std::vector<float> ExtractFeatures();

  // Extract 11-dimensional features for specific rank (Phase 4)
  std::vector<float> ExtractFeaturesForRank(int idx);

  // Extract 7-dimensional traversal window statistics
  std::vector<float> GetTraversalWindowStats(const std::vector<int>& masked_ids);
  std::array<float, 7> GetTraversalWindowStatsArrayFromSortedWindow(
      const std::vector<std::pair<int, float>>& sorted_window,
      const std::vector<int>& masked_ids);
  std::vector<float> GetTraversalWindowStatsFromSortedWindow(
      const std::vector<std::pair<int, float>>& sorted_window,
      const std::vector<int>& masked_ids);

  // Predict with 11-dimensional features
  float PredictWithFeatures(const std::vector<float>& features);
  float PredictWithFeatureArray(const std::array<float, 11>& features);

  // Predict recall for specific rank (Phase 4)
  float PredictRecallForRank(int idx);
  float PredictRecallForRankWithSortedWindow(
      int idx, const std::vector<std::pair<int, float>>& sorted_window);

  // Get prediction interval from interval_table
  std::pair<int, int> GetPredictionInterval(float target_recall) const;

  // Query gt_collected_table (Phase 4)
  float GetRecallFromGtCollectedTable(int collected, int rank);

  // Query gt_cmps_all_table (Phase 4)
  float GetRecallFromGtCmpsAllTable(int rank, int cmps);

  // Scratch buffers reused on the hot prediction path to reduce allocations.
  std::vector<std::pair<int, float>> sorted_window_scratch_;
  std::vector<float> filtered_distances_scratch_;
  std::vector<int> masked_ids_scratch_;
};

} // namespace omega

#endif  // ZVEC_THIRDPARTY_OMEGA_INCLUDE_SEARCH_CONTEXT_H_
