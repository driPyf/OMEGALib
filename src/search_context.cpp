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

namespace omega {

SearchContext::SearchContext(const GBDTModel* model, const ModelTables* tables)
    : model_(model), tables_(tables), feature_extractor_() {}

SearchContext::~SearchContext() = default;

void SearchContext::Reset() {
  state_ = SearchState();
}

void SearchContext::UpdateState(int hops, int comparisons, float distance) {
  state_.curr_hops = hops;
  state_.curr_cmps = comparisons;

  // Update distance window
  state_.distance_window.push_back(distance);

  // Keep window size bounded
  if (state_.distance_window.size() > feature_extractor_.kDefaultWindowSize) {
    state_.distance_window.erase(state_.distance_window.begin());
  }

  // Update dist_1st if this is the first distance or closer
  if (state_.distance_window.size() == 1 || distance < state_.dist_1st) {
    state_.dist_1st = distance;
  }
}

bool SearchContext::ShouldStopEarly(float target_recall) {
  if (!model_ || !tables_) {
    return false;  // No model, can't make decision
  }

  // Predict current score
  float score = PredictScore();

  // Map score to expected recall
  float expected_recall = ScoreToRecall(score);

  // Stop if we've reached target recall
  return expected_recall >= target_recall;
}

int SearchContext::GetOptimalEF(float target_recall, int current_ef) {
  if (!model_ || !tables_) {
    return current_ef;  // No model, use current EF
  }

  // Get EF multiplier for target recall
  float multiplier = GetEFMultiplier(target_recall);

  // Apply multiplier to current EF
  int optimal_ef = static_cast<int>(current_ef * multiplier);

  // Ensure EF is at least 1
  return std::max(1, optimal_ef);
}

float SearchContext::PredictScore() {
  // Extract features from current state
  std::vector<float> features = feature_extractor_.Extract(state_);

  // Convert float features to double for model prediction
  std::vector<double> features_double(features.begin(), features.end());

  // Get model prediction (probability)
  double probability = model_->Predict(features_double.data(), features_double.size());

  // Convert probability to score (0-100 range)
  // This matches the OMEGA paper's approach
  return probability * 100.0f;
}

float SearchContext::ScoreToRecall(float score) {
  if (!tables_ || tables_->threshold_table.empty()) {
    return 0.0f;
  }

  // Threshold table maps int(threshold * 10000) to recall
  return LookupTable(tables_->threshold_table, score, 10000.0f);
}

float SearchContext::GetEFMultiplier(float target_recall) {
  if (!tables_ || tables_->multiplier_table.empty()) {
    return 1.0f;  // Default multiplier
  }

  // Multiplier table maps int(recall * 100) to multiplier
  return LookupTable(tables_->multiplier_table, target_recall, 100.0f);
}

float SearchContext::LookupTable(
    const std::unordered_map<int, float>& table,
    float key, float scale) const {
  if (table.empty()) {
    return 0.0f;
  }

  int scaled_key = static_cast<int>(key * scale);

  // Exact match
  auto it = table.find(scaled_key);
  if (it != table.end()) {
    return it->second;
  }

  // Linear interpolation between nearest neighbors
  int lower_key = -1;
  int upper_key = -1;
  float lower_value = 0.0f;
  float upper_value = 0.0f;

  for (const auto& entry : table) {
    if (entry.first <= scaled_key) {
      if (lower_key == -1 || entry.first > lower_key) {
        lower_key = entry.first;
        lower_value = entry.second;
      }
    }
    if (entry.first >= scaled_key) {
      if (upper_key == -1 || entry.first < upper_key) {
        upper_key = entry.first;
        upper_value = entry.second;
      }
    }
  }

  // If we only have one side, use that value
  if (lower_key == -1) {
    return upper_value;
  }
  if (upper_key == -1) {
    return lower_value;
  }

  // Interpolate
  if (lower_key == upper_key) {
    return lower_value;
  }

  float ratio = static_cast<float>(scaled_key - lower_key) /
                static_cast<float>(upper_key - lower_key);
  return lower_value + ratio * (upper_value - lower_value);
}

} // namespace omega
