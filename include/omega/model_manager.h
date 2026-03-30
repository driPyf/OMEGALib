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

#ifndef ZVEC_THIRDPARTY_OMEGA_INCLUDE_MODEL_MANAGER_H_
#define ZVEC_THIRDPARTY_OMEGA_INCLUDE_MODEL_MANAGER_H_

#include <string>
#include <map>
#include <unordered_map>
#include <memory>
#include "omega/tree_inference.h"

namespace omega {

// Stores the auxiliary tables used by OMEGA for adaptive search.
// These tables are generated during model training and map model outputs
// to search parameters.
struct ModelTables {
  // Maps calibrated confidence (scaled by 10000) to expected recall.
  // Format: threshold_table[int(confidence * 10000)] = recall
  //
  // This table is produced during training via isotonic regression. The goal is
  // calibration, not just ranking: we want the model's confidence score to
  // approximate the true probability that the masked top-1 decision is
  // correct. Weighted BH relies on those per-rank confidence values being
  // probability-like; without calibration, the rank-wise error budget would be
  // systematically misallocated when lifted from top-1 decisions to the final
  // top-k stopping rule.
  //
  // Using std::map for ordered lookup.
  std::map<int, float> threshold_table;

  // Maps recall (scaled by 100) to (initial_interval, min_interval)
  // Format: interval_table[int(recall * 100)] = (initial, min)
  // Using std::map for ordered lookup
  std::map<int, std::pair<int, int>> interval_table;

  // Maps recall (scaled by 100) to EF multiplier
  // Format: multiplier_table[int(recall * 100)] = multiplier
  std::unordered_map<int, float> multiplier_table;

  // Ground-truth collection statistics used for recall lookup tables.
  // 2D table: gt_collected_table[collected][rank] = recall
  // Format in file: "row_index:val1,val2,...,valK\n"
  std::map<int, std::vector<float>> gt_collected_table;

  // Ground-truth comparison statistics used for recall lookup tables.
  // 2D table: gt_cmps_all_table[rank][percentile] = cmps_value
  // Format in file: "row_index:val1,val2,...,val100\n"
  // Contains 100 percentiles (1%, 2%, ..., 100%) for each rank
  std::map<int, std::vector<float>> gt_cmps_all_table;

  ModelTables() = default;
};

// Manages the lifecycle of OMEGA models and their associated tables.
// Handles loading models from disk and providing access to model components.
class ModelManager {
 public:
  ModelManager();
  ~ModelManager();

  // Load model and tables from a directory.
  // Expected files in model_dir:
  // - model.txt: LightGBM model in text format
  // - threshold_table.txt: threshold -> recall mapping
  // - interval_table.txt: recall -> (initial, min) interval
  // - multiplier_table.txt: recall -> multiplier
  // - gt_collected_table.txt: ground truth collection stats
  // - gt_cmps_all_table.txt: ground truth comparison stats
  //
  // Returns true on success, false on failure.
  bool LoadModel(const std::string& model_dir);

  // Get the loaded GBDT model (nullptr if not loaded)
  const GBDTModel* GetModel() const { return model_.get(); }

  // Get the model tables (nullptr if not loaded)
  const ModelTables* GetTables() const { return tables_.get(); }

  // Check if model is successfully loaded
 bool IsLoaded() const { return model_loaded_; }

 private:
  std::unique_ptr<GBDTModel> model_;
  std::unique_ptr<ModelTables> tables_;
  bool model_loaded_;

  // Load individual table files
  bool LoadThresholdTable(const std::string& path);
  bool LoadIntervalTable(const std::string& path);
  bool LoadMultiplierTable(const std::string& path);
  bool LoadGTCollectedTable(const std::string& path);
  bool LoadGTCmpsAllTable(const std::string& path);

  // Helper to parse 2D table line with format "row_index:val1,val2,...,valN"
  bool Parse2DTableLine(const std::string& line, int* row_index,
                       std::vector<float>* values);
};

} // namespace omega

#endif  // ZVEC_THIRDPARTY_OMEGA_INCLUDE_MODEL_MANAGER_H_
