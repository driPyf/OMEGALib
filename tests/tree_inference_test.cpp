// Copyright 2025-present Yifan Peng
//
// Licensed under the MIT License. See the LICENSE file in the project root for
// license terms.

#include "tree_inference.h"

#include <cmath>
#include <fstream>
#include <gtest/gtest.h>

namespace omega {
namespace {

// Test fixture for DecisionTree tests
class DecisionTreeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tree_.Clear();
  }

  DecisionTree tree_;
};

// Test basic tree construction and prediction
TEST_F(DecisionTreeTest, BasicPrediction) {
  // Build a simple tree:
  //       [feature_0 <= 5.0]
  //       /              \
  //   leaf(1.0)      leaf(2.0)

  TreeNode root;
  root.feature_index = 0;
  root.threshold = 5.0;
  root.left_child = 1;
  root.right_child = 2;
  root.is_leaf = false;
  tree_.AddNode(root);

  TreeNode left_leaf;
  left_leaf.leaf_value = 1.0;
  left_leaf.is_leaf = true;
  tree_.AddNode(left_leaf);

  TreeNode right_leaf;
  right_leaf.leaf_value = 2.0;
  right_leaf.is_leaf = true;
  tree_.AddNode(right_leaf);

  // Test predictions
  double features1[] = {3.0};  // <= 5.0, should go left
  EXPECT_DOUBLE_EQ(1.0, tree_.Predict(features1, 1));

  double features2[] = {7.0};  // > 5.0, should go right
  EXPECT_DOUBLE_EQ(2.0, tree_.Predict(features2, 1));

  double features3[] = {5.0};  // == 5.0, should go left
  EXPECT_DOUBLE_EQ(1.0, tree_.Predict(features3, 1));
}

// Test multi-level tree
TEST_F(DecisionTreeTest, MultiLevelTree) {
  // Build a tree:
  //           [feature_0 <= 5.0]
  //           /              \
  //   [feature_1 <= 3.0]   leaf(3.0)
  //     /          \
  // leaf(1.0)   leaf(2.0)

  TreeNode root;
  root.feature_index = 0;
  root.threshold = 5.0;
  root.left_child = 1;
  root.right_child = 2;
  root.is_leaf = false;
  tree_.AddNode(root);

  TreeNode left_split;
  left_split.feature_index = 1;
  left_split.threshold = 3.0;
  left_split.left_child = 3;
  left_split.right_child = 4;
  left_split.is_leaf = false;
  tree_.AddNode(left_split);

  TreeNode right_leaf;
  right_leaf.leaf_value = 3.0;
  right_leaf.is_leaf = true;
  tree_.AddNode(right_leaf);

  TreeNode left_left_leaf;
  left_left_leaf.leaf_value = 1.0;
  left_left_leaf.is_leaf = true;
  tree_.AddNode(left_left_leaf);

  TreeNode left_right_leaf;
  left_right_leaf.leaf_value = 2.0;
  left_right_leaf.is_leaf = true;
  tree_.AddNode(left_right_leaf);

  // Test predictions
  double features1[] = {3.0, 2.0};  // Left, then left
  EXPECT_DOUBLE_EQ(1.0, tree_.Predict(features1, 2));

  double features2[] = {3.0, 4.0};  // Left, then right
  EXPECT_DOUBLE_EQ(2.0, tree_.Predict(features2, 2));

  double features3[] = {7.0, 2.0};  // Right
  EXPECT_DOUBLE_EQ(3.0, tree_.Predict(features3, 2));
}

// Test empty tree
TEST_F(DecisionTreeTest, EmptyTree) {
  double features[] = {1.0};
  EXPECT_DOUBLE_EQ(0.0, tree_.Predict(features, 1));
}

// Test single leaf tree
TEST_F(DecisionTreeTest, SingleLeafTree) {
  TreeNode leaf;
  leaf.leaf_value = 42.0;
  leaf.is_leaf = true;
  tree_.AddNode(leaf);

  double features[] = {1.0, 2.0, 3.0};
  EXPECT_DOUBLE_EQ(42.0, tree_.Predict(features, 3));
}

// Test fixture for GBDTModel tests
class GBDTModelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    model_.Clear();
  }

  // Helper function to create a simple tree
  DecisionTree CreateSimpleTree(double left_value, double right_value) {
    DecisionTree tree;

    TreeNode root;
    root.feature_index = 0;
    root.threshold = 5.0;
    root.left_child = 1;
    root.right_child = 2;
    root.is_leaf = false;
    tree.AddNode(root);

    TreeNode left_leaf;
    left_leaf.leaf_value = left_value;
    left_leaf.is_leaf = true;
    tree.AddNode(left_leaf);

    TreeNode right_leaf;
    right_leaf.leaf_value = right_value;
    right_leaf.is_leaf = true;
    tree.AddNode(right_leaf);

    return tree;
  }

  GBDTModel model_;
};

// Test GBDT prediction with single tree
TEST_F(GBDTModelTest, SingleTreePrediction) {
  model_.SetBaseScore(0.0);
  model_.AddTree(CreateSimpleTree(1.0, 2.0));

  double features1[] = {3.0};
  double raw1 = model_.PredictRaw(features1, 1);
  EXPECT_DOUBLE_EQ(1.0, raw1);

  double prob1 = model_.Predict(features1, 1);
  EXPECT_NEAR(1.0 / (1.0 + std::exp(-1.0)), prob1, 1e-9);

  double features2[] = {7.0};
  double raw2 = model_.PredictRaw(features2, 1);
  EXPECT_DOUBLE_EQ(2.0, raw2);

  double prob2 = model_.Predict(features2, 1);
  EXPECT_NEAR(1.0 / (1.0 + std::exp(-2.0)), prob2, 1e-9);
}

// Test GBDT prediction with multiple trees
TEST_F(GBDTModelTest, MultipleTreesPrediction) {
  model_.SetBaseScore(0.5);
  model_.AddTree(CreateSimpleTree(0.1, 0.2));
  model_.AddTree(CreateSimpleTree(0.3, 0.4));

  double features1[] = {3.0};  // Goes left in both trees
  double raw1 = model_.PredictRaw(features1, 1);
  EXPECT_DOUBLE_EQ(0.5 + 0.1 + 0.3, raw1);  // base + tree1 + tree2

  double features2[] = {7.0};  // Goes right in both trees
  double raw2 = model_.PredictRaw(features2, 1);
  EXPECT_DOUBLE_EQ(0.5 + 0.2 + 0.4, raw2);  // base + tree1 + tree2
}

// Test model save and load
TEST_F(GBDTModelTest, SaveAndLoad) {
  // Create a model
  model_.SetBaseScore(0.5);
  model_.AddTree(CreateSimpleTree(1.0, 2.0));
  model_.AddTree(CreateSimpleTree(3.0, 4.0));

  // Save to file
  std::string temp_file = "/tmp/test_model.txt";
  ASSERT_TRUE(model_.SaveToFile(temp_file));

  // Load from file
  GBDTModel loaded_model;
  ASSERT_TRUE(loaded_model.LoadFromFile(temp_file));

  // Verify loaded model
  EXPECT_DOUBLE_EQ(model_.GetBaseScore(), loaded_model.GetBaseScore());
  EXPECT_EQ(model_.GetTreeCount(), loaded_model.GetTreeCount());

  // Verify predictions match
  double features1[] = {3.0};
  EXPECT_DOUBLE_EQ(model_.PredictRaw(features1, 1),
                   loaded_model.PredictRaw(features1, 1));

  double features2[] = {7.0};
  EXPECT_DOUBLE_EQ(model_.PredictRaw(features2, 1),
                   loaded_model.PredictRaw(features2, 1));

  // Clean up
  std::remove(temp_file.c_str());
}

// Test loading non-existent file
TEST_F(GBDTModelTest, LoadNonExistentFile) {
  EXPECT_FALSE(model_.LoadFromFile("/tmp/non_existent_model.txt"));
}

// Test sigmoid transformation
TEST_F(GBDTModelTest, SigmoidTransformation) {
  model_.SetBaseScore(0.0);

  DecisionTree tree;
  TreeNode leaf;
  leaf.leaf_value = 0.0;
  leaf.is_leaf = true;
  tree.AddNode(leaf);
  model_.AddTree(tree);

  double features[] = {1.0};

  // Test sigmoid(0) = 0.5
  EXPECT_DOUBLE_EQ(0.5, model_.Predict(features, 1));

  // Test sigmoid(large positive) ≈ 1.0
  model_.SetBaseScore(10.0);
  EXPECT_NEAR(1.0, model_.Predict(features, 1), 1e-4);

  // Test sigmoid(large negative) ≈ 0.0
  model_.SetBaseScore(-10.0);
  EXPECT_NEAR(0.0, model_.Predict(features, 1), 1e-4);
}

// Test model with no trees
TEST_F(GBDTModelTest, EmptyModel) {
  model_.SetBaseScore(1.5);

  double features[] = {1.0};
  double raw = model_.PredictRaw(features, 1);
  EXPECT_DOUBLE_EQ(1.5, raw);

  double prob = model_.Predict(features, 1);
  EXPECT_NEAR(1.0 / (1.0 + std::exp(-1.5)), prob, 1e-9);
}

}  // namespace
}  // namespace omega

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
