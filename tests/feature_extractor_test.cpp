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

#include "omega/feature_extractor.h"
#include <gtest/gtest.h>

namespace omega {
namespace {

class FeatureExtractorTest : public ::testing::Test {
 protected:
  FeatureExtractor extractor_;
};

TEST_F(FeatureExtractorTest, BasicExtraction) {
  SearchState state;
  state.curr_hops = 5;
  state.curr_cmps = 100;
  state.dist_1st = 0.5f;
  state.dist_start = 1.0f;
  state.distance_window = {0.5f, 0.6f, 0.7f, 0.8f, 0.9f};

  std::vector<float> features = extractor_.Extract(state);

  ASSERT_EQ(features.size(), 11);
  EXPECT_EQ(features[0], 5.0f);   // curr_hops
  EXPECT_EQ(features[1], 100.0f); // curr_cmps
  EXPECT_EQ(features[2], 0.5f);   // dist_1st
  EXPECT_EQ(features[3], 1.0f);   // dist_start
  // features[4-10] are window statistics
}

TEST_F(FeatureExtractorTest, EmptyDistanceWindow) {
  SearchState state;
  state.curr_hops = 0;
  state.curr_cmps = 0;
  state.dist_1st = 0.0f;
  state.dist_start = 0.0f;
  state.distance_window = {};

  std::vector<float> features = extractor_.Extract(state);

  ASSERT_EQ(features.size(), 11);
  EXPECT_EQ(features[0], 0.0f);
  EXPECT_EQ(features[1], 0.0f);
  EXPECT_EQ(features[2], 0.0f);
  EXPECT_EQ(features[3], 0.0f);
  // Window stats should be zeros
  for (size_t i = 4; i < 11; ++i) {
    EXPECT_EQ(features[i], 0.0f);
  }
}

TEST_F(FeatureExtractorTest, LargeDistanceWindow) {
  SearchState state;
  state.curr_hops = 10;
  state.curr_cmps = 500;
  state.dist_1st = 0.1f;
  state.dist_start = 0.5f;

  // Create a large distance window
  for (int i = 0; i < 200; ++i) {
    state.distance_window.push_back(static_cast<float>(i) * 0.01f);
  }

  std::vector<float> features = extractor_.Extract(state);

  ASSERT_EQ(features.size(), 11);
  EXPECT_EQ(features[0], 10.0f);
  EXPECT_EQ(features[1], 500.0f);
  EXPECT_EQ(features[2], 0.1f);
  EXPECT_EQ(features[3], 0.5f);
  // Window statistics should be computed correctly
  EXPECT_GT(features[4], 0.0f); // mean should be positive
}

TEST_F(FeatureExtractorTest, WindowStatistics) {
  SearchState state;
  state.curr_hops = 1;
  state.curr_cmps = 10;
  state.dist_1st = 1.0f;
  state.dist_start = 1.0f;
  state.distance_window = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

  std::vector<float> features = extractor_.Extract(state);

  ASSERT_EQ(features.size(), 11);

  // Full window mean should be 4.5
  EXPECT_NEAR(features[4], 4.5f, 0.01f);

  // Full window variance should be computed
  EXPECT_GT(features[5], 0.0f);

  // Minimum distance should be 1.0
  EXPECT_EQ(features[10], 1.0f);
}

} // namespace
} // namespace omega
