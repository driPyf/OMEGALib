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
#include <deque>
#include <vector>
#include <set>

namespace omega {

// Manages the state of an adaptive search operation using OMEGA.
// Maintains search statistics, calls the GBDT model for predictions,
// and determines when to stop early based on target recall.
// This is a stateful interface - zvec reports each node visit.
class SearchContext {
 public:
  // Create a search context with parameters
  SearchContext(const GBDTModel* model, const ModelTables* tables,
                float target_recall, int k, int window_size);

  ~SearchContext();

  // Reset the search state for a new query
  void Reset();

  // Report a node visit during search
  void ReportVisit(int node_id, float distance, bool is_in_topk);

  // Report a hop during search
  void ReportHop();

  // Set the distance to the start node (called once at search start)
  void SetDistStart(float dist_start) { dist_start_ = dist_start; }

  // Check if should perform prediction now
  // Based on interval_table and current comparison count
  bool ShouldPredict() const;

  // Predict whether to stop early based on current state and target recall.
  // Returns true if the search should stop, false to continue.
  bool ShouldStopEarly();

  // Get current search statistics
  void GetStats(int* hops, int* comparisons, int* collected_gt) const;

 private:
  const GBDTModel* model_;
  const ModelTables* tables_;

  // Search parameters
  float target_recall_;
  int k_;
  int window_size_;

  // Search state
  std::deque<std::pair<int, float>> traversal_window_;  // (node_id, distance)
  std::set<int> topk_node_ids_;  // Node IDs currently in top-K (for masking)
  int hops_;
  int comparisons_;
  float dist_start_;
  float dist_1st_;
  int collected_gt_;  // Number of ground truth collected
  int next_prediction_cmps_;  // When to predict next

  // Extract 11-dimensional features from current state
  std::vector<float> ExtractFeatures();

  // Extract 7-dimensional traversal window statistics
  std::vector<float> GetTraversalWindowStats(const std::vector<int>& masked_ids);

  // Predict with 11-dimensional features
  float PredictWithFeatures(const std::vector<float>& features);

  // Get prediction interval from interval_table
  std::pair<int, int> GetPredictionInterval(float target_recall);
};

} // namespace omega

#endif  // ZVEC_THIRDPARTY_OMEGA_INCLUDE_SEARCH_CONTEXT_H_
