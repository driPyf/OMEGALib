// Copyright 2025-present Yifan Peng
//
// Licensed under the MIT License. See the LICENSE file in the project root for
// license terms.

#include "omega/model_manager.h"
#include <cmath>
#include <fstream>
#include <sstream>

namespace omega {

ModelManager::ModelManager()
    : model_(nullptr), tables_(nullptr), model_loaded_(false) {}

ModelManager::~ModelManager() = default;

bool ModelManager::LoadModel(const std::string& model_dir) {
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

    std::istringstream iss(line);
    std::string threshold_str;
    std::string recall_str;
    if (!std::getline(iss, threshold_str, ',') || !std::getline(iss, recall_str)) {
      continue;
    }

    try {
      float threshold = std::stof(threshold_str);
      float recall = std::stof(recall_str);
      tables_->threshold_table[static_cast<int>(std::round(threshold * 10000.0f))] = recall;
    } catch (...) {
      continue;
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

    std::istringstream iss(line);
    std::string recall_str;
    std::string initial_str;
    std::string min_str;
    if (!std::getline(iss, recall_str, ',') ||
        !std::getline(iss, initial_str, ',') ||
        !std::getline(iss, min_str)) {
      continue;
    }

    try {
      int key = static_cast<int>(std::round(std::stof(recall_str) * 100.0f));
      int value1 = std::stoi(initial_str);
      int value2 = std::stoi(min_str);
      tables_->interval_table[key] = std::make_pair(value1, value2);
    } catch (...) {
      continue;
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

    std::istringstream iss(line);
    std::string recall_str;
    std::string multiplier_str;
    if (!std::getline(iss, recall_str, ',') || !std::getline(iss, multiplier_str)) {
      continue;
    }

    try {
      int key = static_cast<int>(std::round(std::stof(recall_str) * 100.0f));
      float value = std::stof(multiplier_str);
      tables_->multiplier_table[key] = value;
    } catch (...) {
      continue;
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
