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
#include <unordered_map>
#include <memory>
#include "omega/tree_inference.h"

namespace omega {

// Stores the auxiliary tables used by OMEGA for adaptive search.
// These tables are generated during model training and map model outputs
// to search parameters.
struct ModelTables {
  // Maps threshold value (scaled by 10000) to expected recall
  // Format: threshold_table[int(threshold * 10000)] = recall
  std::unordered_map<int, float> threshold_table;

  // Maps recall (scaled by 100) to (initial_interval, min_interval)
  // Format: interval_table[int(recall * 100)] = (initial, min)
  std::unordered_map<int, std::pair<int, int>> interval_table;

  // Maps recall (scaled by 100) to EF multiplier
  // Format: multiplier_table[int(recall * 100)] = multiplier
  std::unordered_map<int, float> multiplier_table;

  // Ground truth collection statistics
  // Maps some key to collected count
  std::unordered_map<int, int> gt_collected_table;

  // Ground truth comparison statistics
  // Maps some key to total comparisons
  std::unordered_map<int, int> gt_cmps_all_table;

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

  // Get the directory from which the model was loaded
  const std::string& GetModelDir() const { return model_dir_; }

 private:
  std::unique_ptr<GBDTModel> model_;
  std::unique_ptr<ModelTables> tables_;
  bool model_loaded_;
  std::string model_dir_;

  // Load individual table files
  bool LoadThresholdTable(const std::string& path);
  bool LoadIntervalTable(const std::string& path);
  bool LoadMultiplierTable(const std::string& path);
  bool LoadGTCollectedTable(const std::string& path);
  bool LoadGTCmpsAllTable(const std::string& path);

  // Helper to parse a line with format "key,value"
  bool ParseKeyValue(const std::string& line, int* key, float* value);
  bool ParseKeyValue(const std::string& line, int* key, int* value);
  bool ParseKeyValuePair(const std::string& line, int* key,
                        int* value1, int* value2);
};

} // namespace omega

#endif  // ZVEC_THIRDPARTY_OMEGA_INCLUDE_MODEL_MANAGER_H_
