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

#include "omega/omega_trainer.h"
#include <LightGBM/c_api.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <iostream>

namespace omega {

namespace {

// Helper to log with timing
class ScopedTimer {
 public:
  ScopedTimer(const std::string& name, bool verbose)
      : name_(name), verbose_(verbose) {
    start_ = std::chrono::high_resolution_clock::now();
    if (verbose_) {
      std::cout << "[OMEGA] [START] " << name_ << std::endl;
    }
  }
  ~ScopedTimer() {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_).count();
    if (verbose_) {
      std::cout << "[OMEGA] [END]   " << name_ << " | Duration: " << duration << " ms" << std::endl;
    }
  }
 private:
  std::string name_;
  bool verbose_;
  std::chrono::high_resolution_clock::time_point start_;
};

// Helper to check LightGBM return code
#define LGBM_CHECK(call) do { \
  int ret = (call); \
  if (ret != 0) { \
    std::cerr << "[OMEGA] LightGBM error: " << LGBM_GetLastError() << std::endl; \
    return -1; \
  } \
} while (0)

}  // namespace

void OmegaTrainer::PrepareData(
    const std::vector<TrainingRecord>& records,
    std::vector<float>& features,
    std::vector<float>& labels,
    int& num_samples,
    int& num_features) {
  // Features: hops_visited, cmps_visited, dist_1st, dist_start, stat_0..stat_6
  num_features = 11;
  num_samples = static_cast<int>(records.size());

  features.resize(num_samples * num_features);
  labels.resize(num_samples);

  for (int i = 0; i < num_samples; ++i) {
    const auto& r = records[i];
    int offset = i * num_features;

    features[offset + 0] = static_cast<float>(r.hops_visited);
    features[offset + 1] = static_cast<float>(r.cmps_visited);
    features[offset + 2] = r.dist_1st;
    features[offset + 3] = r.dist_start;

    // Copy traversal window stats (7 dimensions)
    for (size_t j = 0; j < 7 && j < r.traversal_window_stats.size(); ++j) {
      features[offset + 4 + j] = r.traversal_window_stats[j];
    }

    labels[i] = static_cast<float>(r.label);
  }
}

int OmegaTrainer::GenerateThresholdTable(
    const std::vector<float>& predictions,
    const std::vector<float>& labels,
    const std::string& output_path) {
  // Sort predictions and corresponding labels
  std::vector<size_t> indices(predictions.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
    return predictions[a] < predictions[b];
  });

  // Compute cumulative positive rate (isotonic-like)
  std::vector<float> sorted_conf(predictions.size());
  std::vector<float> sorted_prob(predictions.size());

  double cumsum = 0;
  for (size_t i = 0; i < indices.size(); ++i) {
    cumsum += labels[indices[i]];
    sorted_conf[i] = predictions[indices[i]];
    sorted_prob[i] = static_cast<float>(cumsum / (i + 1));
  }

  // Apply isotonic regression (pool adjacent violators)
  for (size_t i = 1; i < sorted_prob.size(); ++i) {
    if (sorted_prob[i] < sorted_prob[i-1]) {
      sorted_prob[i] = sorted_prob[i-1];
    }
  }

  // Deduplicate by confidence (quantized to 10000)
  std::ofstream ofs(output_path);
  if (!ofs.is_open()) {
    std::cerr << "[OMEGA] Failed to open threshold table: " << output_path << std::endl;
    return -1;
  }

  int last_quantized = -1;
  for (size_t i = 0; i < sorted_conf.size(); ++i) {
    int quantized = static_cast<int>(std::round(sorted_conf[i] * 10000));
    if (quantized != last_quantized) {
      ofs << std::fixed << std::setprecision(4) << sorted_conf[i] << ","
          << std::setprecision(6) << sorted_prob[i] << "\n";
      last_quantized = quantized;
    }
  }

  return 0;
}

int OmegaTrainer::GenerateGtCmpsAllTable(
    const GtCmpsData& gt_cmps_data,
    const std::string& output_path) {
  std::ofstream ofs(output_path);
  if (!ofs.is_open()) {
    std::cerr << "[OMEGA] Failed to open gt_cmps_all_table: " << output_path << std::endl;
    return -1;
  }

  size_t num_queries = gt_cmps_data.num_queries;
  size_t topk = gt_cmps_data.topk;

  for (size_t rank = 0; rank < topk; ++rank) {
    // Collect all cmps values for this rank
    std::vector<int> cmps_values;
    cmps_values.reserve(num_queries);

    for (size_t q = 0; q < num_queries; ++q) {
      if (rank < gt_cmps_data.gt_cmps[q].size()) {
        cmps_values.push_back(gt_cmps_data.gt_cmps[q][rank]);
      }
    }

    if (cmps_values.empty()) {
      continue;
    }

    // Sort for percentile calculation
    std::sort(cmps_values.begin(), cmps_values.end());

    // Calculate percentiles (1-100)
    ofs << rank << ":";
    for (int pct = 1; pct <= 100; ++pct) {
      size_t idx = static_cast<size_t>((pct / 100.0) * (cmps_values.size() - 1));
      idx = std::min(idx, cmps_values.size() - 1);
      ofs << cmps_values[idx];
      if (pct < 100) ofs << ",";
    }
    ofs << "\n";
  }

  return 0;
}

int OmegaTrainer::GenerateGtCollectedTable(
    const GtCmpsData& gt_cmps_data,
    const std::string& output_path) {
  std::ofstream ofs(output_path);
  if (!ofs.is_open()) {
    std::cerr << "[OMEGA] Failed to open gt_collected_table: " << output_path << std::endl;
    return -1;
  }

  size_t num_queries = gt_cmps_data.num_queries;
  size_t topk = gt_cmps_data.topk;

  for (size_t collected = 0; collected <= topk; ++collected) {
    ofs << collected << ":";

    if (collected == 0) {
      // No GTs collected yet, all probabilities are 0
      for (size_t r = 0; r < topk; ++r) {
        ofs << "0.0";
        if (r < topk - 1) ofs << ",";
      }
    } else {
      for (size_t rank = 0; rank < topk; ++rank) {
        if (rank < collected) {
          // Ranks before "collected" are always found
          ofs << "1.0";
        } else {
          // Compute probability: GT[rank] is collected if cmps[rank] <= cmps[collected-1]
          size_t threshold_rank = collected - 1;
          size_t count_found = 0;

          for (size_t q = 0; q < num_queries; ++q) {
            if (rank < gt_cmps_data.gt_cmps[q].size() &&
                threshold_rank < gt_cmps_data.gt_cmps[q].size()) {
              if (gt_cmps_data.gt_cmps[q][rank] <= gt_cmps_data.gt_cmps[q][threshold_rank]) {
                count_found++;
              }
            }
          }

          double prob = num_queries > 0 ? static_cast<double>(count_found) / num_queries : 0.0;
          ofs << std::fixed << std::setprecision(6) << prob;
        }
        if (rank < topk - 1) ofs << ",";
      }
    }
    ofs << "\n";
  }

  return 0;
}

int OmegaTrainer::GenerateIntervalTable(
    const GtCmpsData& gt_cmps_data,
    const std::string& output_path) {
  std::ofstream ofs(output_path);
  if (!ofs.is_open()) {
    std::cerr << "[OMEGA] Failed to open interval_table: " << output_path << std::endl;
    return -1;
  }

  size_t num_queries = gt_cmps_data.num_queries;

  // Use only the first rank (k=1) for interval table generation
  std::vector<int> gt_cmps_k1;
  gt_cmps_k1.reserve(num_queries);

  for (size_t q = 0; q < num_queries; ++q) {
    if (!gt_cmps_data.gt_cmps[q].empty()) {
      gt_cmps_k1.push_back(gt_cmps_data.gt_cmps[q][0]);
    }
  }

  if (gt_cmps_k1.empty()) {
    std::cerr << "[OMEGA] No gt_cmps data for interval table" << std::endl;
    return -1;
  }

  std::sort(gt_cmps_k1.begin(), gt_cmps_k1.end());

  for (int recall_pct = 0; recall_pct <= 100; ++recall_pct) {
    double recall = recall_pct / 100.0;
    size_t idx = static_cast<size_t>(recall * (gt_cmps_k1.size() - 1));
    idx = std::min(idx, gt_cmps_k1.size() - 1);

    int interval = gt_cmps_k1[idx];
    int initial_interval = std::max(interval / 2, 1);
    int min_interval = std::max(interval / 10, 1);

    ofs << std::fixed << std::setprecision(2) << recall << ","
        << initial_interval << "," << min_interval << "\n";
  }

  return 0;
}

int OmegaTrainer::TrainModel(
    const std::vector<TrainingRecord>& training_records,
    const GtCmpsData& gt_cmps_data,
    const OmegaTrainerOptions& options) {

  if (training_records.empty()) {
    std::cerr << "[OMEGA] Training records are empty" << std::endl;
    return -1;
  }

  if (options.output_dir.empty()) {
    std::cerr << "[OMEGA] Output directory is empty" << std::endl;
    return -1;
  }

  auto total_start = std::chrono::high_resolution_clock::now();

  // Step 1: Prepare training data
  std::vector<float> features, labels;
  int num_samples, num_features;

  {
    ScopedTimer timer("PrepareData", options.verbose);
    PrepareData(training_records, features, labels, num_samples, num_features);
  }

  // Count positive and negative samples
  int n_positive = 0, n_negative = 0;
  for (float l : labels) {
    if (l > 0.5f) n_positive++;
    else n_negative++;
  }

  if (n_positive == 0 || n_negative == 0) {
    std::cerr << "[OMEGA] Invalid label distribution: " << n_positive
              << " positive, " << n_negative << " negative" << std::endl;
    return -1;
  }

  double scale_pos_weight = static_cast<double>(n_negative) / n_positive;

  if (options.verbose) {
    std::cout << "[OMEGA] Training samples: " << num_samples
              << " (" << n_positive << " positive, " << n_negative << " negative)" << std::endl;
    std::cout << "[OMEGA] scale_pos_weight: " << scale_pos_weight << std::endl;
  }

  // Step 2: Split into train/test (80/20)
  int train_size = static_cast<int>(num_samples * 0.8);
  int test_size = num_samples - train_size;

  // Step 3: Create LightGBM datasets
  DatasetHandle train_data = nullptr;
  DatasetHandle test_data = nullptr;
  BoosterHandle booster = nullptr;

  {
    ScopedTimer timer("CreateDataset", options.verbose);

    // Build parameters string
    std::ostringstream params_ss;
    params_ss << "num_threads=" << options.num_threads << " ";
    params_ss << "verbosity=-1 ";  // Force silent mode
    params_ss << "force_row_wise=true";
    std::string params_str = params_ss.str();

    // Create training dataset
    LGBM_CHECK(LGBM_DatasetCreateFromMat(
        features.data(),
        C_API_DTYPE_FLOAT32,
        train_size,
        num_features,
        1,  // is_row_major
        params_str.c_str(),
        nullptr,
        &train_data));

    // Set training labels
    LGBM_CHECK(LGBM_DatasetSetField(
        train_data,
        "label",
        labels.data(),
        train_size,
        C_API_DTYPE_FLOAT32));

    // Create test dataset using training data as reference
    LGBM_CHECK(LGBM_DatasetCreateFromMat(
        features.data() + train_size * num_features,
        C_API_DTYPE_FLOAT32,
        test_size,
        num_features,
        1,  // is_row_major
        params_str.c_str(),
        train_data,  // reference
        &test_data));

    // Set test labels
    LGBM_CHECK(LGBM_DatasetSetField(
        test_data,
        "label",
        labels.data() + train_size,
        test_size,
        C_API_DTYPE_FLOAT32));
  }

  // Step 4: Create booster and train
  {
    ScopedTimer timer("TrainLightGBM", options.verbose);

    // Build booster parameters
    std::ostringstream booster_params;
    booster_params << "task=train ";
    booster_params << "boosting_type=gbdt ";
    booster_params << "objective=binary ";
    booster_params << "metric=binary_logloss ";
    booster_params << "num_leaves=" << options.num_leaves << " ";
    booster_params << "learning_rate=" << options.learning_rate << " ";
    booster_params << "feature_fraction=1.0 ";
    booster_params << "bagging_fraction=1.0 ";
    booster_params << "bagging_freq=0 ";
    booster_params << "boost_from_average=false ";
    booster_params << "verbosity=-1 ";  // Force silent mode
    booster_params << "num_threads=" << options.num_threads << " ";
    booster_params << "scale_pos_weight=" << scale_pos_weight << " ";
    booster_params << "force_row_wise=true";

    std::string booster_params_str = booster_params.str();

    if (options.verbose) {
      std::cout << "[OMEGA] About to create booster with params: " << booster_params_str << std::endl;
    }

    LGBM_CHECK(LGBM_BoosterCreate(train_data, booster_params_str.c_str(), &booster));

    if (options.verbose) {
      std::cout << "[OMEGA] Booster created successfully" << std::endl;
    }

    // Add validation data
    LGBM_CHECK(LGBM_BoosterAddValidData(booster, test_data));

    if (options.verbose) {
      std::cout << "[OMEGA] Starting training for " << options.num_iterations << " iterations..." << std::endl;
    }

    // Train for num_iterations
    for (int i = 0; i < options.num_iterations; ++i) {
      if (options.verbose && i % 10 == 0) {
        std::cout << "[OMEGA] Training iteration " << i << "/" << options.num_iterations << std::endl;
      }
      int is_finished = 0;
      LGBM_CHECK(LGBM_BoosterUpdateOneIter(booster, &is_finished));

      if (is_finished) {
        if (options.verbose) {
          std::cout << "[OMEGA] Training finished early at iteration " << i << std::endl;
        }
        break;
      }
    }

    if (options.verbose) {
      std::cout << "[OMEGA] Training completed" << std::endl;
    }
  }

  // Step 5: Save model
  std::string model_path = options.output_dir + "/model.txt";
  {
    ScopedTimer timer("SaveModel", options.verbose);
    LGBM_CHECK(LGBM_BoosterSaveModel(booster, 0, -1, C_API_FEATURE_IMPORTANCE_SPLIT, model_path.c_str()));
  }

  // Step 6: Get predictions on test set for threshold table
  std::vector<float> test_predictions;  // Declare outside scope for use in Step 7
  {
    ScopedTimer timer("Predict", options.verbose);

    // LightGBM C API always outputs double, so use double array directly
    std::vector<double> test_preds_double(test_size);
    int64_t out_len = 0;

    LGBM_CHECK(LGBM_BoosterPredictForMat(
        booster,
        features.data() + train_size * num_features,
        C_API_DTYPE_FLOAT32,
        test_size,
        num_features,
        1,  // is_row_major
        C_API_PREDICT_NORMAL,
        0,  // start_iteration
        -1, // num_iteration (all)
        "",
        &out_len,
        test_preds_double.data()));

    // Convert double to float for threshold table generation
    test_predictions.resize(test_size);
    for (int i = 0; i < test_size; ++i) {
      test_predictions[i] = static_cast<float>(test_preds_double[i]);
    }
  }

  // Step 7: Generate threshold table
  {
    ScopedTimer timer("GenerateThresholdTable", options.verbose);
    std::vector<float> test_labels(labels.begin() + train_size, labels.end());
    std::string threshold_path = options.output_dir + "/threshold_table.txt";
    if (GenerateThresholdTable(test_predictions, test_labels, threshold_path) != 0) {
      std::cerr << "[OMEGA] Failed to generate threshold table" << std::endl;
      // Continue anyway
    }
  }

  // Step 8: Generate tables from gt_cmps data
  if (gt_cmps_data.num_queries > 0) {
    {
      ScopedTimer timer("GenerateGtCmpsAllTable", options.verbose);
      std::string path = options.output_dir + "/gt_cmps_all_table.txt";
      GenerateGtCmpsAllTable(gt_cmps_data, path);
    }

    {
      ScopedTimer timer("GenerateGtCollectedTable", options.verbose);
      std::string path = options.output_dir + "/gt_collected_table.txt";
      GenerateGtCollectedTable(gt_cmps_data, path);
    }

    {
      ScopedTimer timer("GenerateIntervalTable", options.verbose);
      std::string path = options.output_dir + "/interval_table.txt";
      GenerateIntervalTable(gt_cmps_data, path);
    }
  }

  // Cleanup
  if (booster) LGBM_BoosterFree(booster);
  if (test_data) LGBM_DatasetFree(test_data);
  if (train_data) LGBM_DatasetFree(train_data);

  auto total_end = std::chrono::high_resolution_clock::now();
  auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();

  if (options.verbose) {
    std::cout << "[OMEGA] Training completed successfully in " << total_ms << " ms" << std::endl;
    std::cout << "[OMEGA] Model saved to: " << model_path << std::endl;
  }

  return 0;
}

}  // namespace omega

