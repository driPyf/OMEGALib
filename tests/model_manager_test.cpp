// Copyright 2025-present Yifan Peng
//
// Licensed under the MIT License. See the LICENSE file in the project root for
// license terms.

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include "omega/model_manager.h"
#include "omega/tree_inference.h"

namespace omega {
namespace {

namespace fs = std::filesystem;

class ModelManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    test_dir_ = fs::temp_directory_path() /
                fs::path("omega_model_manager_test_" +
                         std::to_string(::testing::UnitTest::GetInstance()
                                            ->random_seed()));
    fs::create_directories(test_dir_);
  }

  void TearDown() override { fs::remove_all(test_dir_); }

  void WriteFile(const std::string& name, const std::string& contents) {
    std::ofstream out(test_dir_ / name);
    out << contents;
  }

  void WriteMinimalModel() {
    GBDTModel model;
    model.SetBaseScore(0.0);
    DecisionTree tree;
    TreeNode leaf;
    leaf.is_leaf = true;
    leaf.leaf_value = 0.75;
    tree.AddNode(leaf);
    model.AddTree(tree);
    ASSERT_TRUE(model.SaveToFile((test_dir_ / "model.txt").string()));
  }

  fs::path test_dir_;
};

TEST_F(ModelManagerTest, LoadModelSucceedsWithOptionalTablesPresent) {
  WriteMinimalModel();
  WriteFile("threshold_table.txt", "0.8000,0.91\n0.9000,0.96\n");
  WriteFile("interval_table.txt", "0.92,60,15\n");
  WriteFile("multiplier_table.txt", "0.92,1.50\n");
  WriteFile("gt_collected_table.txt", "1:0.80,0.88\n2:0.90,0.95\n");
  WriteFile("gt_cmps_all_table.txt", "1:10,20,30\n2:15,25,35\n");

  ModelManager manager;
  ASSERT_TRUE(manager.LoadModel(test_dir_.string()));
  ASSERT_TRUE(manager.IsLoaded());
  ASSERT_NE(manager.GetModel(), nullptr);
  ASSERT_NE(manager.GetTables(), nullptr);

  const auto* tables = manager.GetTables();
  ASSERT_EQ(tables->threshold_table.at(8000), 0.91f);
  ASSERT_EQ(tables->interval_table.at(92), std::make_pair(60, 15));
  ASSERT_FLOAT_EQ(tables->multiplier_table.at(92), 1.5f);
  ASSERT_EQ(tables->gt_collected_table.at(2).size(), 2U);
  ASSERT_EQ(tables->gt_cmps_all_table.at(1).size(), 3U);
}

TEST_F(ModelManagerTest, MissingOptionalTablesStillLoadsModel) {
  WriteMinimalModel();

  ModelManager manager;
  ASSERT_TRUE(manager.LoadModel(test_dir_.string()));
  ASSERT_TRUE(manager.IsLoaded());
  ASSERT_NE(manager.GetModel(), nullptr);
  ASSERT_NE(manager.GetTables(), nullptr);
  EXPECT_TRUE(manager.GetTables()->threshold_table.empty());
  EXPECT_TRUE(manager.GetTables()->interval_table.empty());
}

TEST_F(ModelManagerTest, MissingModelFileFailsAndLeavesManagerUnloaded) {
  WriteFile("threshold_table.txt", "0.8000,0.91\n");

  ModelManager manager;
  EXPECT_FALSE(manager.LoadModel(test_dir_.string()));
  EXPECT_FALSE(manager.IsLoaded());
  EXPECT_EQ(manager.GetModel(), nullptr);
}

}  // namespace
}  // namespace omega
