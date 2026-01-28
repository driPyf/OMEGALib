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

#include "tree_inference.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace omega {

double DecisionTree::Predict(const double* features,
                              int32_t num_features) const {
  if (nodes_.empty()) {
    return 0.0;
  }

  int32_t node_idx = 0;
  while (!nodes_[node_idx].is_leaf) {
    const TreeNode& node = nodes_[node_idx];

    // Validate feature index
    if (node.feature_index < 0 || node.feature_index >= num_features) {
      throw std::runtime_error("Invalid feature index: " +
                               std::to_string(node.feature_index));
    }

    // Navigate to left or right child based on threshold
    if (features[node.feature_index] <= node.threshold) {
      node_idx = node.left_child;
    } else {
      node_idx = node.right_child;
    }

    // Validate child index
    if (node_idx < 0 || node_idx >= static_cast<int32_t>(nodes_.size())) {
      throw std::runtime_error("Invalid child node index: " +
                               std::to_string(node_idx));
    }
  }

  return nodes_[node_idx].leaf_value;
}

int32_t DecisionTree::AddNode(const TreeNode& node) {
  int32_t index = static_cast<int32_t>(nodes_.size());
  nodes_.push_back(node);
  return index;
}

double GBDTModel::Predict(const double* features, int32_t num_features) const {
  double raw_score = PredictRaw(features, num_features);
  // Apply sigmoid transformation: 1 / (1 + exp(-x))
  return 1.0 / (1.0 + std::exp(-raw_score));
}

double GBDTModel::PredictRaw(const double* features,
                              int32_t num_features) const {
  double sum = base_score_;
  for (const auto& tree : trees_) {
    sum += tree.Predict(features, num_features);
  }
  return sum;
}

bool GBDTModel::LoadFromFile(const std::string& file_path) {
  std::ifstream file(file_path);
  if (!file.is_open()) {
    return false;
  }

  Clear();

  std::string line;
  DecisionTree* current_tree = nullptr;
  int32_t tree_count = 0;

  while (std::getline(file, line)) {
    // Skip empty lines and comments
    if (line.empty() || line[0] == '#') {
      continue;
    }

    // Parse base_score
    if (line.find("base_score=") == 0) {
      base_score_ = std::stod(line.substr(11));
      continue;
    }

    // Parse tree_count
    if (line.find("tree_count=") == 0) {
      tree_count = std::stoi(line.substr(11));
      continue;
    }

    // Start of a new tree
    if (line.find("Tree=") == 0) {
      trees_.emplace_back();
      current_tree = &trees_.back();
      continue;
    }

    // Parse tree nodes
    if (current_tree != nullptr && line.find("split_feature=") == 0) {
      TreeNode node;
      std::istringstream iss(line);
      std::string token;

      // Parse split_feature
      std::getline(iss, token, '=');
      std::getline(iss, token, ' ');
      node.feature_index = std::stoi(token);

      // Parse threshold
      std::getline(iss, token, '=');
      std::getline(iss, token, ' ');
      node.threshold = std::stod(token);

      // Parse left_child
      std::getline(iss, token, '=');
      std::getline(iss, token, ' ');
      node.left_child = std::stoi(token);

      // Parse right_child
      std::getline(iss, token, '=');
      std::getline(iss, token, ' ');
      node.right_child = std::stoi(token);

      node.is_leaf = false;
      current_tree->AddNode(node);
      continue;
    }

    // Parse leaf nodes
    if (current_tree != nullptr && line.find("leaf_value=") == 0) {
      TreeNode node;
      node.leaf_value = std::stod(line.substr(11));
      node.is_leaf = true;
      current_tree->AddNode(node);
      continue;
    }
  }

  file.close();

  // Validate tree count
  if (tree_count > 0 && static_cast<int32_t>(trees_.size()) != tree_count) {
    Clear();
    return false;
  }

  return !trees_.empty();
}

bool GBDTModel::SaveToFile(const std::string& file_path) const {
  std::ofstream file(file_path);
  if (!file.is_open()) {
    return false;
  }

  // Write header
  file << "# OMEGA GBDT Model\n";
  file << "base_score=" << base_score_ << "\n";
  file << "tree_count=" << trees_.size() << "\n\n";

  // Write each tree
  for (size_t tree_idx = 0; tree_idx < trees_.size(); ++tree_idx) {
    file << "Tree=" << tree_idx << "\n";
    const DecisionTree& tree = trees_[tree_idx];

    for (int32_t node_idx = 0; node_idx < tree.GetNodeCount(); ++node_idx) {
      const TreeNode& node = tree.GetNode(node_idx);

      if (node.is_leaf) {
        file << "leaf_value=" << node.leaf_value << "\n";
      } else {
        file << "split_feature=" << node.feature_index
             << " threshold=" << node.threshold
             << " left_child=" << node.left_child
             << " right_child=" << node.right_child << "\n";
      }
    }
    file << "\n";
  }

  file.close();
  return true;
}

}  // namespace omega
