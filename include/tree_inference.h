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

#ifndef ZVEC_THIRDPARTY_OMEGA_INCLUDE_TREE_INFERENCE_H_
#define ZVEC_THIRDPARTY_OMEGA_INCLUDE_TREE_INFERENCE_H_

#include <cstdint>
#include <string>
#include <vector>

namespace omega {

// Represents a single node in a decision tree.
// Each node is either:
// - A split node: has feature_index and threshold, routes to left/right child
// - A leaf node: has leaf_value, no children
struct TreeNode {
  // Feature index to split on (0-based). Only valid for split nodes.
  int32_t feature_index;

  // Threshold value for splitting. Only valid for split nodes.
  // If feature_value <= threshold, go to left child; otherwise right child.
  double threshold;

  // Index of left child node in the nodes array. -1 if leaf node.
  int32_t left_child;

  // Index of right child node in the nodes array. -1 if leaf node.
  int32_t right_child;

  // Leaf value for prediction. Only valid for leaf nodes.
  double leaf_value;

  // Whether this node is a leaf node.
  bool is_leaf;

  TreeNode()
      : feature_index(-1),
        threshold(0.0),
        left_child(-1),
        right_child(-1),
        leaf_value(0.0),
        is_leaf(false) {}
};

// Represents a single decision tree.
class DecisionTree {
 public:
  DecisionTree() = default;
  ~DecisionTree() = default;

  // Predicts the output for given features.
  // Args:
  //   features: Array of feature values
  //   num_features: Number of features
  // Returns:
  //   Predicted value (leaf value)
  double Predict(const double* features, int32_t num_features) const;

  // Adds a node to the tree.
  // Args:
  //   node: The tree node to add
  // Returns:
  //   Index of the added node
  int32_t AddNode(const TreeNode& node);

  // Gets the number of nodes in the tree.
  int32_t GetNodeCount() const { return static_cast<int32_t>(nodes_.size()); }

  // Gets a node by index.
  const TreeNode& GetNode(int32_t index) const { return nodes_[index]; }

  // Gets mutable access to a node by index.
  TreeNode* GetMutableNode(int32_t index) { return &nodes_[index]; }

  // Clears all nodes in the tree.
  void Clear() { nodes_.clear(); }

 private:
  std::vector<TreeNode> nodes_;
};

// Represents a Gradient Boosted Decision Tree (GBDT) model.
// The model consists of multiple decision trees and a base score.
// Prediction is computed as: sigmoid(base_score + sum of tree predictions)
class GBDTModel {
 public:
  GBDTModel() : base_score_(0.0) {}
  ~GBDTModel() = default;

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

  // Loads model from a text file.
  // The file format is compatible with LightGBM's text model format.
  // Args:
  //   file_path: Path to the model file
  // Returns:
  //   true if successful, false otherwise
  bool LoadFromFile(const std::string& file_path);

  // Saves model to a text file.
  // Args:
  //   file_path: Path to save the model
  // Returns:
  //   true if successful, false otherwise
  bool SaveToFile(const std::string& file_path) const;

  // Gets the number of trees in the model.
  int32_t GetTreeCount() const { return static_cast<int32_t>(trees_.size()); }

  // Gets a tree by index.
  const DecisionTree& GetTree(int32_t index) const { return trees_[index]; }

  // Gets mutable access to a tree by index.
  DecisionTree* GetMutableTree(int32_t index) { return &trees_[index]; }

  // Adds a tree to the model.
  // Args:
  //   tree: The decision tree to add
  void AddTree(const DecisionTree& tree) { trees_.push_back(tree); }

  // Sets the base score.
  void SetBaseScore(double base_score) { base_score_ = base_score; }

  // Gets the base score.
  double GetBaseScore() const { return base_score_; }

  // Clears all trees and resets base score.
  void Clear() {
    trees_.clear();
    base_score_ = 0.0;
  }

 private:
  std::vector<DecisionTree> trees_;
  double base_score_;
};

}  // namespace omega

#endif  // ZVEC_THIRDPARTY_OMEGA_INCLUDE_TREE_INFERENCE_H_
