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

#include "omega/model_manager.h"
#include <fstream>
#include <sstream>

namespace omega {

ModelManager::ModelManager()
    : model_(nullptr), tables_(nullptr), model_loaded_(false) {}

ModelManager::~ModelManager() = default;

bool ModelManager::LoadModel(const std::string& model_dir) {
  model_dir_ = model_dir;
  model_loaded_ = false;

  // Create new model and tables
  model_ = std::make_unique<GBDTModel>();
  tables_ = std::make_unique<ModelTables>();

  // Load GBDT model
  std::string model_path = model_dir + "/model.txt";
  if (!model_->LoadFromFile(model_path)) {
    return false;
  }

  // Load tables (some may be optional)
  std::string threshold_path = model_dir + "/threshold_table.txt";
  if (!LoadThresholdTable(threshold_path)) {
    // Threshold table is optional, continue
  }

  std::string interval_path = model_dir + "/interval_table.txt";
  if (!LoadIntervalTable(interval_path)) {
    // Interval table is optional, continue
  }

  std::string multiplier_path = model_dir + "/multiplier_table.txt";
  if (!LoadMultiplierTable(multiplier_path)) {
    // Multiplier table is optional, continue
  }

  std::string gt_collected_path = model_dir + "/gt_collected_table.txt";
  if (!LoadGTCollectedTable(gt_collected_path)) {
    // GT collected table is optional, continue
  }

  std::string gt_cmps_path = model_dir + "/gt_cmps_all_table.txt";
  if (!LoadGTCmpsAllTable(gt_cmps_path)) {
    // GT cmps table is optional, continue
  }

  model_loaded_ = true;
  return true;
}

bool ModelManager::LoadThresholdTable(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    return false;
  }

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') {
      continue;  // Skip empty lines and comments
    }

    int key;
    float value;
    if (ParseKeyValue(line, &key, &value)) {
      tables_->threshold_table[key] = value;
    }
  }

  return true;
}

bool ModelManager::LoadIntervalTable(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    return false;
  }

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }

    int key, value1, value2;
    if (ParseKeyValuePair(line, &key, &value1, &value2)) {
      tables_->interval_table[key] = std::make_pair(value1, value2);
    }
  }

  return true;
}

bool ModelManager::LoadMultiplierTable(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    return false;
  }

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }

    int key;
    float value;
    if (ParseKeyValue(line, &key, &value)) {
      tables_->multiplier_table[key] = value;
    }
  }

  return true;
}

bool ModelManager::LoadGTCollectedTable(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    return false;
  }

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }

    // Parse 2D format: "row_index:val1,val2,...,valN"
    int row_index;
    std::vector<float> values;
    if (Parse2DTableLine(line, &row_index, &values)) {
      tables_->gt_collected_table[row_index] = std::move(values);
    }
  }

  return true;
}

bool ModelManager::LoadGTCmpsAllTable(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    return false;
  }

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }

    // Parse 2D format: "row_index:val1,val2,...,valN"
    int row_index;
    std::vector<float> values;
    if (Parse2DTableLine(line, &row_index, &values)) {
      tables_->gt_cmps_all_table[row_index] = std::move(values);
    }
  }

  return true;
}

bool ModelManager::ParseKeyValue(const std::string& line,
                                 int* key, float* value) {
  std::istringstream iss(line);
  std::string key_str, value_str;

  if (!std::getline(iss, key_str, ',')) {
    return false;
  }
  if (!std::getline(iss, value_str)) {
    return false;
  }

  try {
    *key = std::stoi(key_str);
    *value = std::stof(value_str);
    return true;
  } catch (...) {
    return false;
  }
}

bool ModelManager::ParseKeyValue(const std::string& line,
                                 int* key, int* value) {
  std::istringstream iss(line);
  std::string key_str, value_str;

  if (!std::getline(iss, key_str, ',')) {
    return false;
  }
  if (!std::getline(iss, value_str)) {
    return false;
  }

  try {
    *key = std::stoi(key_str);
    *value = std::stoi(value_str);
    return true;
  } catch (...) {
    return false;
  }
}

bool ModelManager::ParseKeyValuePair(const std::string& line, int* key,
                                    int* value1, int* value2) {
  std::istringstream iss(line);
  std::string key_str, value1_str, value2_str;

  if (!std::getline(iss, key_str, ',')) {
    return false;
  }
  if (!std::getline(iss, value1_str, ',')) {
    return false;
  }
  if (!std::getline(iss, value2_str)) {
    return false;
  }

  try {
    *key = std::stoi(key_str);
    *value1 = std::stoi(value1_str);
    *value2 = std::stoi(value2_str);
    return true;
  } catch (...) {
    return false;
  }
}

bool ModelManager::Parse2DTableLine(const std::string& line, int* row_index,
                                   std::vector<float>* values) {
  // Parse format: "row_index:val1,val2,...,valN"
  size_t colon_pos = line.find(':');
  if (colon_pos == std::string::npos) {
    return false;
  }

  // Parse row index
  std::string row_index_str = line.substr(0, colon_pos);
  try {
    *row_index = std::stoi(row_index_str);
  } catch (...) {
    return false;
  }

  // Parse comma-separated values
  std::string values_str = line.substr(colon_pos + 1);
  std::istringstream iss(values_str);
  std::string value_str;

  values->clear();
  while (std::getline(iss, value_str, ',')) {
    try {
      values->push_back(std::stof(value_str));
    } catch (...) {
      return false;
    }
  }

  return !values->empty();
}

} // namespace omega

