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
#include <limits>
#include <mutex>
#include <unordered_map>
#include "omega/tree_inference.h"

namespace omega {

namespace {

struct TimingStatsState {
  std::mutex mu;
  std::vector<std::pair<std::string, int64_t>> ordered_stats;
  std::unordered_map<std::string, size_t> index_by_name;
};

TimingStatsState& GetTimingStatsState() {
  static TimingStatsState state;
  return state;
}

void RecordTimingStat(const std::string& name, int64_t duration_ms) {
  auto& state = GetTimingStatsState();
  std::lock_guard<std::mutex> lock(state.mu);
  auto it = state.index_by_name.find(name);
  if (it == state.index_by_name.end()) {
    state.index_by_name[name] = state.ordered_stats.size();
    state.ordered_stats.emplace_back(name, duration_ms);
  } else {
    state.ordered_stats[it->second].second = duration_ms;
  }
}

void WriteTimingStatsJson(
    const std::string& output_path,
    const std::vector<std::pair<std::string, int64_t>>& stats) {
  std::ofstream ofs(output_path);
  if (!ofs.is_open()) {
    return;
  }
  ofs << "{\n";
  for (size_t i = 0; i < stats.size(); ++i) {
    ofs << "  \"" << stats[i].first << "\": " << stats[i].second;
    if (i + 1 < stats.size()) {
      ofs << ",";
    }
    ofs << "\n";
  }
  ofs << "}\n";
}

void WriteMetricsJson(
    const std::string& output_path,
    const std::vector<std::string>& metric_names,
    const std::vector<std::pair<int, std::vector<double>>>& train_history,
    const std::vector<std::pair<int, std::vector<double>>>& valid_history,
    int best_iteration,
    double best_valid_metric) {
  std::ofstream ofs(output_path);
  if (!ofs.is_open()) {
    return;
  }

  ofs << "{\n";
  ofs << "  \"metric_names\": [";
  for (size_t i = 0; i < metric_names.size(); ++i) {
    ofs << "\"" << metric_names[i] << "\"";
    if (i + 1 < metric_names.size()) {
      ofs << ", ";
    }
  }
  ofs << "],\n";
  ofs << "  \"best_iteration\": " << best_iteration << ",\n";
  ofs << "  \"best_valid_metric\": " << std::fixed << std::setprecision(8)
      << best_valid_metric << ",\n";

  auto write_history = [&ofs](
                           const char* key,
                           const std::vector<std::pair<int, std::vector<double>>>& history) {
    ofs << "  \"" << key << "\": [\n";
    for (size_t i = 0; i < history.size(); ++i) {
      ofs << "    {\"iteration\": " << history[i].first << ", \"values\": [";
      for (size_t j = 0; j < history[i].second.size(); ++j) {
        ofs << std::fixed << std::setprecision(8) << history[i].second[j];
        if (j + 1 < history[i].second.size()) {
          ofs << ", ";
        }
      }
      ofs << "]}";
      if (i + 1 < history.size()) {
        ofs << ",";
      }
      ofs << "\n";
    }
    ofs << "  ]";
  };

  write_history("train_history", train_history);
  ofs << ",\n";
  write_history("valid_history", valid_history);
  ofs << "\n}\n";
}

double ClampProbability(double p) {
  constexpr double kEps = 1e-15;
  return std::min(1.0 - kEps, std::max(kEps, p));
}

double BinaryLogLoss(const std::vector<float>& predictions,
                     const std::vector<float>& labels) {
  if (predictions.empty() || predictions.size() != labels.size()) {
    return 0.0;
  }

  double loss = 0.0;
  for (size_t i = 0; i < predictions.size(); ++i) {
    const double p = ClampProbability(predictions[i]);
    const double y = labels[i] > 0.5f ? 1.0 : 0.0;
    loss += -(y * std::log(p) + (1.0 - y) * std::log(1.0 - p));
  }
  return loss / static_cast<double>(predictions.size());
}

void WritePredictionTraceCsv(const std::string& output_path,
                             const std::vector<int>& query_ids,
                             const std::vector<float>& predictions,
                             const std::vector<float>& labels) {
  if (predictions.size() != labels.size() || predictions.size() != query_ids.size()) {
    return;
  }

  std::ofstream ofs(output_path);
  if (!ofs.is_open()) {
    return;
  }

  ofs << "query_id,prediction,label\n";
  for (size_t i = 0; i < predictions.size(); ++i) {
    ofs << query_ids[i] << ","
        << std::fixed << std::setprecision(6) << predictions[i] << ","
        << static_cast<int>(labels[i] > 0.5f) << "\n";
  }
}

void WriteCalibrationBucketsCsv(const std::string& output_path,
                                const std::vector<float>& predictions,
                                const std::vector<float>& labels,
                                int bucket_count,
                                bool verbose,
                                const char* split_name) {
  if (predictions.empty() || predictions.size() != labels.size() || bucket_count <= 0) {
    return;
  }

  struct BucketStats {
    int count = 0;
    double pred_sum = 0.0;
    double label_sum = 0.0;
  };

  std::vector<BucketStats> buckets(static_cast<size_t>(bucket_count));
  for (size_t i = 0; i < predictions.size(); ++i) {
    const double p = std::min(0.999999, std::max(0.0, static_cast<double>(predictions[i])));
    int idx = static_cast<int>(p * bucket_count);
    if (idx >= bucket_count) {
      idx = bucket_count - 1;
    }
    BucketStats& bucket = buckets[static_cast<size_t>(idx)];
    bucket.count += 1;
    bucket.pred_sum += p;
    bucket.label_sum += (labels[i] > 0.5f ? 1.0 : 0.0);
  }

  std::ofstream ofs(output_path);
  if (!ofs.is_open()) {
    return;
  }

  ofs << "bucket_start,bucket_end,count,avg_prediction,positive_rate\n";
  for (int i = 0; i < bucket_count; ++i) {
    const BucketStats& bucket = buckets[static_cast<size_t>(i)];
    const double begin = static_cast<double>(i) / bucket_count;
    const double end = static_cast<double>(i + 1) / bucket_count;
    const double avg_prediction =
        bucket.count > 0 ? bucket.pred_sum / bucket.count : 0.0;
    const double positive_rate =
        bucket.count > 0 ? bucket.label_sum / bucket.count : 0.0;
    ofs << std::fixed << std::setprecision(6)
        << begin << "," << end << "," << bucket.count << ","
        << avg_prediction << "," << positive_rate << "\n";
  }

  if (verbose) {
    std::cout << "[OMEGA] " << split_name << " calibration buckets:" << std::endl;
    for (int i = 0; i < bucket_count; ++i) {
      const BucketStats& bucket = buckets[static_cast<size_t>(i)];
      if (bucket.count == 0) {
        continue;
      }
      const double begin = static_cast<double>(i) / bucket_count;
      const double end = static_cast<double>(i + 1) / bucket_count;
      const double avg_prediction = bucket.pred_sum / bucket.count;
      const double positive_rate = bucket.label_sum / bucket.count;
      std::cout << "[OMEGA]   [" << std::fixed << std::setprecision(2)
                << begin << ", " << end << "): count=" << bucket.count
                << " avg_pred=" << std::setprecision(6) << avg_prediction
                << " positive_rate=" << positive_rate << std::endl;
    }
  }
}

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
    RecordTimingStat(name_, duration);
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

void OmegaTrainer::ResetTimingStats() {
  auto& state = GetTimingStatsState();
  std::lock_guard<std::mutex> lock(state.mu);
  state.ordered_stats.clear();
  state.index_by_name.clear();
}

OmegaTrainer::TimingStats OmegaTrainer::ConsumeTimingStats() {
  auto& state = GetTimingStatsState();
  std::lock_guard<std::mutex> lock(state.mu);
  TimingStats timings = std::move(state.ordered_stats);
  state.ordered_stats.clear();
  state.index_by_name.clear();
  return timings;
}

void OmegaTrainer::PrepareData(
    const std::vector<TrainingRecord>& records,
    std::vector<int>& query_ids,
    std::vector<float>& features,
    std::vector<float>& labels,
    int& num_samples,
    int& num_features) {
  // Features: hops_visited, cmps_visited, dist_1st, dist_start, stat_0..stat_6
  num_features = 11;
  num_samples = static_cast<int>(records.size());

  query_ids.resize(num_samples);
  features.resize(num_samples * num_features);
  labels.resize(num_samples);

  for (int i = 0; i < num_samples; ++i) {
    const auto& r = records[i];
    int offset = i * num_features;

    query_ids[i] = r.query_id;
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
  struct Block {
    size_t begin;
    size_t end;
    double sum;
    size_t count;
  };

  if (predictions.empty() || predictions.size() != labels.size()) {
    return -1;
  }

  // Sort predictions and corresponding labels
  std::vector<size_t> indices(predictions.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
    return predictions[a] < predictions[b];
  });

  std::vector<float> sorted_conf(predictions.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    sorted_conf[i] = predictions[indices[i]];
  }
  std::vector<float> isotonic_prob(predictions.size(), 0.0f);

  // Pool adjacent violators on labels sorted by prediction score.
  std::vector<Block> blocks;
  blocks.reserve(indices.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    blocks.push_back(Block{i, i + 1, static_cast<double>(labels[indices[i]]), 1});
    while (blocks.size() >= 2) {
      const auto& curr = blocks.back();
      const auto& prev = blocks[blocks.size() - 2];
      double curr_avg = curr.sum / static_cast<double>(curr.count);
      double prev_avg = prev.sum / static_cast<double>(prev.count);
      if (prev_avg <= curr_avg) {
        break;
      }

      Block merged{
          prev.begin,
          curr.end,
          prev.sum + curr.sum,
          prev.count + curr.count,
      };
      blocks.pop_back();
      blocks.pop_back();
      blocks.push_back(merged);
    }
  }

  for (const auto& block : blocks) {
    float block_avg = static_cast<float>(block.sum / static_cast<double>(block.count));
    for (size_t i = block.begin; i < block.end; ++i) {
      isotonic_prob[i] = block_avg;
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
          << std::setprecision(6) << isotonic_prob[i] << "\n";
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

  // Reference format includes a leading zero row for collected_gt = 0 / rank 0.
  ofs << 0 << ":";
  for (int pct = 1; pct <= 100; ++pct) {
    ofs << 0;
    if (pct < 100) ofs << ",";
  }
  ofs << "\n";

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
    ofs << (rank + 1) << ":";
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
  ResetTimingStats();
  auto subset_gt_cmps = [](const GtCmpsData& src,
                           const std::vector<int>& query_ids_subset) -> GtCmpsData {
    GtCmpsData dst;
    dst.topk = src.topk;
    dst.gt_cmps.reserve(query_ids_subset.size());
    dst.total_cmps.reserve(query_ids_subset.size());
    for (int query_id : query_ids_subset) {
      if (query_id < 0 || query_id >= static_cast<int>(src.gt_cmps.size()) ||
          query_id >= static_cast<int>(src.total_cmps.size())) {
        continue;
      }
      dst.gt_cmps.push_back(src.gt_cmps[query_id]);
      dst.total_cmps.push_back(src.total_cmps[query_id]);
    }
    dst.num_queries = dst.gt_cmps.size();
    return dst;
  };

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
  std::vector<int> query_ids;
  std::vector<float> features, labels;
  int num_samples, num_features;

  {
    ScopedTimer timer("PrepareData", options.verbose);
    PrepareData(training_records, query_ids, features, labels, num_samples, num_features);
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

  // Step 2: Split into train/test (80/20) by query_id without shuffling.
  std::vector<int> ordered_query_ids;
  ordered_query_ids.reserve(query_ids.size());
  bool query_ids_non_decreasing = true;
  int prev_query_id = std::numeric_limits<int>::min();
  for (int query_id : query_ids) {
    if (query_id < prev_query_id) {
      query_ids_non_decreasing = false;
    }
    if (ordered_query_ids.empty() || ordered_query_ids.back() != query_id) {
      ordered_query_ids.push_back(query_id);
    }
    prev_query_id = query_id;
  }

  if (!query_ids_non_decreasing) {
    std::cerr << "[OMEGA] Training records are not grouped by query_id; "
                 "query-level split requires non-decreasing query ids"
              << std::endl;
    return -1;
  }

  size_t train_query_count = ordered_query_ids.size() * 8 / 10;
  if (train_query_count == 0 || train_query_count >= ordered_query_ids.size()) {
    std::cerr << "[OMEGA] Invalid query-level split with "
              << ordered_query_ids.size() << " unique queries" << std::endl;
    return -1;
  }

  std::vector<int> train_query_ids(
      ordered_query_ids.begin(), ordered_query_ids.begin() + train_query_count);
  std::vector<int> test_query_ids(
      ordered_query_ids.begin() + train_query_count, ordered_query_ids.end());

  int split_query_id = test_query_ids.front();
  auto split_it = std::lower_bound(query_ids.begin(), query_ids.end(), split_query_id);
  if (split_it == query_ids.end()) {
    std::cerr << "[OMEGA] Failed to find split point for query_id="
              << split_query_id << std::endl;
    return -1;
  }

  int train_size = static_cast<int>(std::distance(query_ids.begin(), split_it));
  int test_size = num_samples - train_size;
  if (train_size == 0 || test_size == 0) {
    std::cerr << "[OMEGA] Empty train/test split after query-level partition" << std::endl;
    return -1;
  }

  GtCmpsData gt_cmps_test = subset_gt_cmps(gt_cmps_data, test_query_ids);

  // Step 3: Create LightGBM datasets
  DatasetHandle train_data = nullptr;
  DatasetHandle test_data = nullptr;
  BoosterHandle booster = nullptr;
  std::vector<std::string> eval_metric_names;
  std::vector<std::pair<int, std::vector<double>>> train_eval_history;
  std::vector<std::pair<int, std::vector<double>>> valid_eval_history;
  int best_iteration = -1;
  double best_valid_metric = std::numeric_limits<double>::infinity();

  {
    ScopedTimer timer("CreateDataset", options.verbose);

    // Build parameters string
    std::ostringstream params_ss;
    params_ss << "num_threads=" << options.num_threads << " ";
    params_ss << "seed=" << options.seed << " ";
    params_ss << "data_random_seed=" << options.seed << " ";
    params_ss << "feature_fraction_seed=" << options.seed << " ";
    params_ss << "bagging_seed=" << options.seed << " ";
    params_ss << "extra_seed=" << options.seed << " ";
    params_ss << "verbosity=-1 ";  // Force silent mode
    params_ss << "deterministic=" << (options.deterministic ? "true" : "false") << " ";
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
        features.data() + static_cast<size_t>(train_size) * num_features,
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
    booster_params << "seed=" << options.seed << " ";
    booster_params << "data_random_seed=" << options.seed << " ";
    booster_params << "feature_fraction_seed=" << options.seed << " ";
    booster_params << "bagging_seed=" << options.seed << " ";
    booster_params << "extra_seed=" << options.seed << " ";
    booster_params << "deterministic="
                   << (options.deterministic ? "true" : "false") << " ";
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

    int eval_count = 0;
    LGBM_CHECK(LGBM_BoosterGetEvalCounts(booster, &eval_count));
    if (eval_count > 0) {
      eval_metric_names.resize(static_cast<size_t>(eval_count));
      std::vector<std::vector<char>> eval_name_storage(
          static_cast<size_t>(eval_count), std::vector<char>(64, '\0'));
      std::vector<char*> eval_name_ptrs;
      eval_name_ptrs.reserve(static_cast<size_t>(eval_count));
      for (auto& storage : eval_name_storage) {
        eval_name_ptrs.push_back(storage.data());
      }
      int out_len = 0;
      size_t out_buffer_len = 0;
      LGBM_CHECK(LGBM_BoosterGetEvalNames(
          booster, eval_count, &out_len, 64, &out_buffer_len, eval_name_ptrs.data()));
      for (int i = 0; i < eval_count; ++i) {
        eval_metric_names[static_cast<size_t>(i)] = eval_name_ptrs[static_cast<size_t>(i)];
      }
    }

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

      if (!eval_metric_names.empty() &&
          ((i + 1) % 10 == 0 || i + 1 == options.num_iterations || is_finished)) {
        std::vector<double> train_eval(eval_metric_names.size(), 0.0);
        std::vector<double> valid_eval(eval_metric_names.size(), 0.0);
        int train_eval_len = 0;
        int valid_eval_len = 0;
        LGBM_CHECK(LGBM_BoosterGetEval(
            booster, 0, &train_eval_len, train_eval.data()));
        LGBM_CHECK(LGBM_BoosterGetEval(
            booster, 1, &valid_eval_len, valid_eval.data()));
        train_eval.resize(static_cast<size_t>(train_eval_len));
        valid_eval.resize(static_cast<size_t>(valid_eval_len));
        train_eval_history.emplace_back(i + 1, train_eval);
        valid_eval_history.emplace_back(i + 1, valid_eval);

        if (!valid_eval.empty() && valid_eval[0] < best_valid_metric) {
          best_valid_metric = valid_eval[0];
          best_iteration = i + 1;
        }

        if (options.verbose && !train_eval.empty() && !valid_eval.empty()) {
          std::cout << "[OMEGA] iter=" << (i + 1)
                    << " train_" << eval_metric_names[0] << "="
                    << std::fixed << std::setprecision(8) << train_eval[0]
                    << " valid_" << eval_metric_names[0] << "="
                    << valid_eval[0] << std::endl;
        }
      }

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

  // Step 6: Get predictions on train/test sets for diagnostics and threshold table
  std::vector<float> train_predictions;
  std::vector<float> test_predictions;
  {
    ScopedTimer timer("Predict", options.verbose);

    std::vector<double> train_preds_double(train_size);
    int64_t train_out_len = 0;
    LGBM_CHECK(LGBM_BoosterPredictForMat(
        booster,
        features.data(),
        C_API_DTYPE_FLOAT32,
        train_size,
        num_features,
        1,
        C_API_PREDICT_NORMAL,
        0,
        -1,
        "",
        &train_out_len,
        train_preds_double.data()));
    train_predictions.resize(train_size);
    for (int i = 0; i < train_size; ++i) {
      train_predictions[static_cast<size_t>(i)] =
          static_cast<float>(train_preds_double[static_cast<size_t>(i)]);
    }

    std::vector<double> test_preds_double(test_size);
    int64_t out_len = 0;

    LGBM_CHECK(LGBM_BoosterPredictForMat(
        booster,
        features.data() + static_cast<size_t>(train_size) * num_features,
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

    test_predictions.resize(test_size);
    for (int i = 0; i < test_size; ++i) {
      test_predictions[i] = static_cast<float>(test_preds_double[i]);
    }
  }

  std::vector<float> train_labels(labels.begin(), labels.begin() + train_size);
  std::vector<float> test_labels(labels.begin() + train_size, labels.end());
  std::vector<int> train_query_ids(query_ids.begin(), query_ids.begin() + train_size);
  std::vector<int> test_query_ids(query_ids.begin() + train_size, query_ids.end());

  const double train_logloss = BinaryLogLoss(train_predictions, train_labels);
  const double valid_logloss = BinaryLogLoss(test_predictions, test_labels);
  if (options.verbose) {
    std::cout << "[OMEGA] final_train_logloss=" << std::fixed << std::setprecision(8)
              << train_logloss << " final_valid_logloss=" << valid_logloss;
    if (best_iteration > 0) {
      std::cout << " best_valid_iteration=" << best_iteration
                << " best_valid_metric=" << best_valid_metric;
    }
    std::cout << std::endl;
  }
  WriteMetricsJson(options.output_dir + "/lightgbm_training_metrics.json",
                   eval_metric_names,
                   train_eval_history,
                   valid_eval_history,
                   best_iteration,
                   std::isfinite(best_valid_metric) ? best_valid_metric
                                                    : valid_logloss);
  WritePredictionTraceCsv(options.output_dir + "/train_predictions.csv",
                          train_query_ids, train_predictions, train_labels);
  WritePredictionTraceCsv(options.output_dir + "/valid_predictions.csv",
                          test_query_ids, test_predictions, test_labels);
  WriteCalibrationBucketsCsv(options.output_dir + "/train_calibration_buckets.csv",
                             train_predictions, train_labels, 20,
                             options.verbose, "train");
  WriteCalibrationBucketsCsv(options.output_dir + "/valid_calibration_buckets.csv",
                             test_predictions, test_labels, 20,
                             options.verbose, "valid");

  // Step 7: Generate threshold table
  {
    ScopedTimer timer("GenerateThresholdTable", options.verbose);
    std::string threshold_path = options.output_dir + "/threshold_table.txt";
    if (GenerateThresholdTable(test_predictions, test_labels, threshold_path) != 0) {
      std::cerr << "[OMEGA] Failed to generate threshold table" << std::endl;
      // Continue anyway
    }
  }

  // Step 8: Generate tables from gt_cmps data
  if (gt_cmps_test.num_queries > 0) {
    {
      ScopedTimer timer("GenerateGtCmpsAllTable", options.verbose);
      std::string path = options.output_dir + "/gt_cmps_all_table.txt";
      GenerateGtCmpsAllTable(gt_cmps_test, path);
    }

    {
      ScopedTimer timer("GenerateGtCollectedTable", options.verbose);
      std::string path = options.output_dir + "/gt_collected_table.txt";
      GenerateGtCollectedTable(gt_cmps_test, path);
    }

    {
      ScopedTimer timer("GenerateIntervalTable", options.verbose);
      std::string path = options.output_dir + "/interval_table.txt";
      GenerateIntervalTable(gt_cmps_test, path);
    }
  }

  // Cleanup
  if (booster) LGBM_BoosterFree(booster);
  if (test_data) LGBM_DatasetFree(test_data);
  if (train_data) LGBM_DatasetFree(train_data);

  auto total_end = std::chrono::high_resolution_clock::now();
  auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
  RecordTimingStat("TrainModel [TOTAL]", total_ms);
  WriteTimingStatsJson(options.output_dir + "/lightgbm_training_timing.json",
                       ConsumeTimingStats());

  if (options.verbose) {
    std::cout << "[OMEGA] Training completed successfully in " << total_ms << " ms" << std::endl;
    std::cout << "[OMEGA] Model saved to: " << model_path << std::endl;
  }

  return 0;
}

}  // namespace omega
