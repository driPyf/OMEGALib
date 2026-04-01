// Copyright 2025-present Yifan Peng
//
// Licensed under the MIT License. See the LICENSE file in the project root for
// license terms.

#pragma once

#include "omega/search_context.h"
#include <string>
#include <vector>
#include <cstdint>
#include <utility>

namespace omega {

/**
 * @brief Ground truth cmps data for table generation
 */
struct GtCmpsData {
  size_t num_queries = 0;
  size_t topk = 0;
  std::vector<std::vector<int>> gt_cmps;  // [query_id][rank] = cmps
  std::vector<int> total_cmps;            // total cmps per query
};

/**
 * @brief Configuration options for OMEGA model training
 */
struct OmegaTrainerOptions {
  std::string output_dir;
  int num_iterations = 100;
  int num_leaves = 31;
  double learning_rate = 0.1;
  int num_threads = 0;  // 0 = use caller/default strategy
  int seed = 42;
  bool deterministic = true;
  bool verbose = false;
  size_t topk = 100;
};

/**
 * @brief OMEGA model trainer using LightGBM C API
 *
 * This class trains a LightGBM binary classifier directly in C++,
 * eliminating the need for Python subprocess and CSV serialization.
 */
class OmegaTrainer {
 public:
  using TimingStats = std::vector<std::pair<std::string, int64_t>>;

  /**
   * @brief Train OMEGA model from collected training records
   *
   * @param training_records Training data collected from searches
   * @param gt_cmps_data Ground truth cmps data for table generation
   * @param options Training configuration
   * @return 0 on success, -1 on failure
   */
  static int TrainModel(
      const std::vector<TrainingRecord>& training_records,
      const GtCmpsData& gt_cmps_data,
      const OmegaTrainerOptions& options);

  static void ResetTimingStats();

  static TimingStats ConsumeTimingStats();

 private:
  /**
   * @brief Prepare feature matrix and labels from training records
   */
  static void PrepareData(
      const std::vector<TrainingRecord>& records,
      std::vector<int>& query_ids,
      std::vector<float>& features,
      std::vector<float>& labels,
      int& num_samples,
      int& num_features);

  /**
   * @brief Generate threshold table using isotonic regression approximation
   */
  static int GenerateThresholdTable(
      const std::vector<float>& predictions,
      const std::vector<float>& labels,
      const std::string& output_path);

  /**
   * @brief Generate gt_cmps_all_table
   */
  static int GenerateGtCmpsAllTable(
      const GtCmpsData& gt_cmps_data,
      const std::string& output_path);

  /**
   * @brief Generate gt_collected_table
   */
  static int GenerateGtCollectedTable(
      const GtCmpsData& gt_cmps_data,
      const std::string& output_path);

  /**
   * @brief Generate interval_table
   */
  static int GenerateIntervalTable(
      const GtCmpsData& gt_cmps_data,
      const std::string& output_path);
};

}  // namespace omega
