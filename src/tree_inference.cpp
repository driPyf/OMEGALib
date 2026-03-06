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

#include "omega/tree_inference.h"
#include <cmath>
#include <cstdio>
#include <cstring>

namespace omega {

GBDTModel::GBDTModel() : booster_(nullptr), num_features_(0), num_iterations_(0) {}

GBDTModel::~GBDTModel() {
  if (booster_ != nullptr) {
    LGBM_BoosterFree(booster_);
    booster_ = nullptr;
  }
}

GBDTModel::GBDTModel(GBDTModel&& other) noexcept
    : booster_(other.booster_),
      num_features_(other.num_features_),
      num_iterations_(other.num_iterations_) {
  other.booster_ = nullptr;
  other.num_features_ = 0;
  other.num_iterations_ = 0;
}

GBDTModel& GBDTModel::operator=(GBDTModel&& other) noexcept {
  if (this != &other) {
    if (booster_ != nullptr) {
      LGBM_BoosterFree(booster_);
    }
    booster_ = other.booster_;
    num_features_ = other.num_features_;
    num_iterations_ = other.num_iterations_;
    other.booster_ = nullptr;
    other.num_features_ = 0;
    other.num_iterations_ = 0;
  }
  return *this;
}

double GBDTModel::Predict(const double* features, int32_t num_features) const {
  double raw_score = PredictRaw(features, num_features);
  // Apply sigmoid transformation
  return 1.0 / (1.0 + std::exp(-raw_score));
}

double GBDTModel::PredictRaw(const double* features, int32_t num_features) const {
  if (booster_ == nullptr) {
    return 0.0;
  }

  int64_t out_len = 0;
  double out_result = 0.0;

  // Use LGBM_BoosterPredictForMatSingleRow for single sample prediction
  int ret = LGBM_BoosterPredictForMatSingleRow(
      booster_,
      features,
      C_API_DTYPE_FLOAT64,
      num_features,
      1,  // is_row_major
      C_API_PREDICT_RAW_SCORE,  // predict_type: raw score (before sigmoid)
      0,  // start_iteration
      -1,  // num_iteration: -1 means use all iterations
      "",  // parameter (empty string for defaults)
      &out_len,
      &out_result);

  if (ret != 0) {
    fprintf(stderr, "LightGBM prediction error: %s\n", LGBM_GetLastError());
    return 0.0;
  }

  return out_result;
}

bool GBDTModel::LoadFromFile(const std::string& file_path) {
  // Free existing booster if any
  if (booster_ != nullptr) {
    LGBM_BoosterFree(booster_);
    booster_ = nullptr;
  }

  int ret = LGBM_BoosterCreateFromModelfile(file_path.c_str(), &num_iterations_, &booster_);
  if (ret != 0) {
    fprintf(stderr, "Failed to load LightGBM model from %s: %s\n",
            file_path.c_str(), LGBM_GetLastError());
    booster_ = nullptr;
    return false;
  }

  // Get number of features
  ret = LGBM_BoosterGetNumFeature(booster_, &num_features_);
  if (ret != 0) {
    fprintf(stderr, "Failed to get number of features: %s\n", LGBM_GetLastError());
    LGBM_BoosterFree(booster_);
    booster_ = nullptr;
    return false;
  }

  return true;
}

bool GBDTModel::SaveToFile(const std::string& file_path) const {
  if (booster_ == nullptr) {
    fprintf(stderr, "No model loaded, cannot save\n");
    return false;
  }

  int ret = LGBM_BoosterSaveModel(
      booster_,
      0,  // start_iteration
      -1,  // num_iteration: -1 means all iterations
      C_API_FEATURE_IMPORTANCE_SPLIT,  // feature_importance_type
      file_path.c_str());

  if (ret != 0) {
    fprintf(stderr, "Failed to save LightGBM model to %s: %s\n",
            file_path.c_str(), LGBM_GetLastError());
    return false;
  }

  return true;
}

int32_t GBDTModel::GetTreeCount() const {
  return num_iterations_;
}

}  // namespace omega
