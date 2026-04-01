// Copyright 2025-present Yifan Peng
//
// Licensed under the MIT License. See the LICENSE file in the project root for
// license terms.

#include "omega/tree_inference.h"
#include <cmath>
#include <cstdio>

namespace omega {

GBDTModel::GBDTModel()
    : boosting_(nullptr),
      tree_early_stop_(
          LightGBM::CreatePredictionEarlyStopInstance(
              std::string("none"), LightGBM::PredictionEarlyStopConfig())),
      num_features_(0),
      num_iterations_(0) {}

GBDTModel::~GBDTModel() {
  if (boosting_ != nullptr) {
    delete boosting_;
    boosting_ = nullptr;
  }
}

GBDTModel::GBDTModel(GBDTModel&& other) noexcept
    : boosting_(other.boosting_),
      tree_early_stop_(std::move(other.tree_early_stop_)),
      num_features_(other.num_features_),
      num_iterations_(other.num_iterations_) {
  other.boosting_ = nullptr;
  other.num_features_ = 0;
  other.num_iterations_ = 0;
}

GBDTModel& GBDTModel::operator=(GBDTModel&& other) noexcept {
  if (this != &other) {
    if (boosting_ != nullptr) {
      delete boosting_;
    }
    boosting_ = other.boosting_;
    tree_early_stop_ = std::move(other.tree_early_stop_);
    num_features_ = other.num_features_;
    num_iterations_ = other.num_iterations_;
    other.boosting_ = nullptr;
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
  if (boosting_ == nullptr || features == nullptr || num_features <= 0) {
    return 0.0;
  }

  double out_result = 0.0;
  boosting_->PredictRaw(features, &out_result, &tree_early_stop_);
  return out_result;
}

bool GBDTModel::LoadFromFile(const std::string& file_path) {
  if (boosting_ != nullptr) {
    delete boosting_;
    boosting_ = nullptr;
  }

  boosting_ = LightGBM::Boosting::CreateBoosting(
      std::string("gbdt"), file_path.c_str(), std::string("cpu"), 0);
  if (boosting_ == nullptr) {
    fprintf(stderr, "Failed to create LightGBM Boosting from %s\n",
            file_path.c_str());
    return false;
  }

  num_iterations_ = boosting_->NumberOfTotalModel();
  boosting_->InitPredict(0, num_iterations_, false);
  num_features_ = boosting_->MaxFeatureIdx() + 1;

  return true;
}

bool GBDTModel::SaveToFile(const std::string& file_path) const {
  if (boosting_ == nullptr) {
    fprintf(stderr, "No model loaded, cannot save\n");
    return false;
  }

  if (!boosting_->SaveModelToFile(
          0, -1, 0, file_path.c_str())) {
    fprintf(stderr, "Failed to save LightGBM model to %s\n",
            file_path.c_str());
    return false;
  }

  return true;
}

int32_t GBDTModel::GetTreeCount() const {
  return num_iterations_;
}

}  // namespace omega
