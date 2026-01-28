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

namespace omega {

// Manages the state of an adaptive search operation using OMEGA.
// Maintains search statistics, calls the GBDT model for predictions,
// and determines when to stop early based on target recall.
class SearchContext {
 public:
  // Create a search context with a model and tables.
  // model and tables must remain valid for the lifetime of this object.
  SearchContext(const GBDTModel* model, const ModelTables* tables);

  ~SearchContext();

  // Reset the search state for a new query
  void Reset();

  // Update search state with new information
  void UpdateState(int hops, int comparisons, float distance);

  // Predict whether to stop early based on current state and target recall.
  // Returns true if the search should stop, false to continue.
  bool ShouldStopEarly(float target_recall);

  // Get the optimal EF value for the current search state.
  // Uses the model prediction and tables to determine the best EF.
  int GetOptimalEF(float target_recall, int current_ef);

  // Get current search state (for debugging/logging)
  const SearchState& GetState() const { return state_; }

 private:
  const GBDTModel* model_;
  const ModelTables* tables_;
  FeatureExtractor feature_extractor_;
  SearchState state_;

  // Predict the model score for current state
  float PredictScore();

  // Map model score to expected recall using threshold table
  float ScoreToRecall(float score);

  // Get EF multiplier for target recall using multiplier table
  float GetEFMultiplier(float target_recall);

  // Lookup value in table with linear interpolation
  float LookupTable(const std::unordered_map<int, float>& table,
                   float key, float scale) const;
};

} // namespace omega

#endif  // ZVEC_THIRDPARTY_OMEGA_INCLUDE_SEARCH_CONTEXT_H_
