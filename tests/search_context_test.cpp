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

#include "omega/search_context.h"
#include <gtest/gtest.h>

namespace omega {
namespace {

class SearchContextTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a simple model for testing
    model_ = std::make_unique<GBDTModel>();
    model_->SetBaseScore(0.0);

    // Add a simple tree: if feature[0] > 5, predict 0.8, else 0.2
    DecisionTree tree;
    TreeNode root;
    root.is_leaf = false;
    root.feature_index = 0;
    root.threshold = 5.0;
    root.left_child = 1;
    root.right_child = 2;
    tree.AddNode(root);

    TreeNode left_leaf;
    left_leaf.is_leaf = true;
    left_leaf.leaf_value = 0.2;
    tree.AddNode(left_leaf);

    TreeNode right_leaf;
    right_leaf.is_leaf = true;
    right_leaf.leaf_value = 0.8;
    tree.AddNode(right_leaf);

    model_->AddTree(tree);

    // Create tables
    tables_ = std::make_unique<ModelTables>();

    // Simple threshold table: score 0.2 -> recall 0.5, score 0.8 -> recall 0.95
    tables_->threshold_table[2000] = 0.5f;   // 0.2 * 10000
    tables_->threshold_table[8000] = 0.95f;  // 0.8 * 10000

    // Simple multiplier table: recall 0.5 -> 1.0x, recall 0.95 -> 0.5x
    tables_->multiplier_table[50] = 1.0f;   // 0.5 * 100
    tables_->multiplier_table[95] = 0.5f;   // 0.95 * 100
  }

  std::unique_ptr<GBDTModel> model_;
  std::unique_ptr<ModelTables> tables_;
};

TEST_F(SearchContextTest, BasicCreation) {
  SearchContext context(model_.get(), tables_.get());
  EXPECT_EQ(context.GetState().curr_hops, 0);
  EXPECT_EQ(context.GetState().curr_cmps, 0);
}

TEST_F(SearchContextTest, UpdateState) {
  SearchContext context(model_.get(), tables_.get());

  context.UpdateState(5, 100, 0.5f);

  const auto& state = context.GetState();
  EXPECT_EQ(state.curr_hops, 5);
  EXPECT_EQ(state.curr_cmps, 100);
  EXPECT_EQ(state.distance_window.size(), 1);
  EXPECT_EQ(state.distance_window[0], 0.5f);
}

TEST_F(SearchContextTest, Reset) {
  SearchContext context(model_.get(), tables_.get());

  context.UpdateState(5, 100, 0.5f);
  context.Reset();

  const auto& state = context.GetState();
  EXPECT_EQ(state.curr_hops, 0);
  EXPECT_EQ(state.curr_cmps, 0);
  EXPECT_EQ(state.distance_window.size(), 0);
}

TEST_F(SearchContextTest, ShouldStopEarlyWithoutModel) {
  SearchContext context(nullptr, nullptr);
  EXPECT_FALSE(context.ShouldStopEarly(0.95f));
}

TEST_F(SearchContextTest, GetOptimalEFWithoutModel) {
  SearchContext context(nullptr, nullptr);
  int ef = context.GetOptimalEF(0.95f, 100);
  EXPECT_EQ(ef, 100);  // Should return current EF
}

TEST_F(SearchContextTest, GetOptimalEFWithModel) {
  SearchContext context(model_.get(), tables_.get());

  // Update state to trigger high prediction
  context.UpdateState(10, 200, 0.5f);

  int ef = context.GetOptimalEF(0.95f, 100);
  // With high recall target and multiplier 0.5, EF should be reduced
  EXPECT_LE(ef, 100);
  EXPECT_GT(ef, 0);
}

TEST_F(SearchContextTest, MultipleUpdates) {
  SearchContext context(model_.get(), tables_.get());

  for (int i = 0; i < 10; ++i) {
    context.UpdateState(i, i * 10, static_cast<float>(i) * 0.1f);
  }

  const auto& state = context.GetState();
  EXPECT_EQ(state.curr_hops, 9);
  EXPECT_EQ(state.curr_cmps, 90);
  EXPECT_EQ(state.distance_window.size(), 10);
}

} // namespace
} // namespace omega
