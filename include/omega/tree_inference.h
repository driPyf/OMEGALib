// Copyright 2025-present Yifan Peng
//
// Licensed under the MIT License. See the LICENSE file in the project root for
// license terms.

#ifndef ZVEC_THIRDPARTY_OMEGA_INCLUDE_TREE_INFERENCE_H_
#define ZVEC_THIRDPARTY_OMEGA_INCLUDE_TREE_INFERENCE_H_

#include <LightGBM/boosting.h>
#include <LightGBM/prediction_early_stop.h>
#include <cstdint>
#include <memory>
#include <string>

namespace omega {

// Wrapper around LightGBM Boosting for GBDT model inference.
// Matches the reference implementation's C++ prediction path.
class GBDTModel {
 public:
  GBDTModel();
  ~GBDTModel();

  // Non-copyable
  GBDTModel(const GBDTModel&) = delete;
  GBDTModel& operator=(const GBDTModel&) = delete;

  // Movable
  GBDTModel(GBDTModel&& other) noexcept;
  GBDTModel& operator=(GBDTModel&& other) noexcept;

  // Predicts the probability for given features.
  // Args:
  //   features: Array of feature values
  //   num_features: Number of features
  // Returns:
  //   Predicted probability (after sigmoid transformation)
  double Predict(const double* features, int32_t num_features) const;

  // Predicts the raw score (before sigmoid) for given features.
  // Args:
  //   features: Array of feature values
  //   num_features: Number of features
  // Returns:
  //   Raw prediction score
  double PredictRaw(const double* features, int32_t num_features) const;

  // Loads model from a text file (LightGBM format).
  // Args:
  //   file_path: Path to the model file
  // Returns:
  //   true if successful, false otherwise
  bool LoadFromFile(const std::string& file_path);

  // Saves model to a text file (LightGBM format).
  // Args:
  //   file_path: Path to save the model
  // Returns:
  //   true if successful, false otherwise
  bool SaveToFile(const std::string& file_path) const;

  // Gets the number of trees in the model.
  int32_t GetTreeCount() const;

  // Check if model is loaded
  bool IsLoaded() const { return boosting_ != nullptr; }

 private:
  LightGBM::Boosting* boosting_;
  LightGBM::PredictionEarlyStopInstance tree_early_stop_;
  int num_features_;
  int num_iterations_;
};

}  // namespace omega

#endif  // ZVEC_THIRDPARTY_OMEGA_INCLUDE_TREE_INFERENCE_H_
