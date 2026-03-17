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

#include "omega/ground_truth.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <vector>

// Eigen for fast matrix operations (cross-platform, header-only)
#include <Eigen/Core>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace omega {

namespace {

// Type aliases for Eigen matrices
using MatrixXfRowMajor = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using VectorXf = Eigen::VectorXf;
using MapConstMatrix = Eigen::Map<const MatrixXfRowMajor>;
using MapVector = Eigen::Map<VectorXf>;
using MapConstVector = Eigen::Map<const VectorXf>;

// Pair of (index, distance) for priority queue
using IndexDistPair = std::pair<size_t, float>;

// Max heap comparator (we want to keep the smallest distances)
struct MaxHeapCmp {
  bool operator()(const IndexDistPair& a, const IndexDistPair& b) const {
    return a.second < b.second;
  }
};

using MaxHeap = std::priority_queue<IndexDistPair, std::vector<IndexDistPair>, MaxHeapCmp>;

}  // namespace

std::vector<std::vector<uint64_t>> ComputeGroundTruth(
    const float* base_vectors,
    const float* query_vectors,
    size_t num_base,
    size_t num_queries,
    size_t dim,
    size_t k,
    MetricType metric,
    bool exclude_self,
    const std::vector<uint64_t>& query_base_indices) {

  std::vector<std::vector<uint64_t>> ground_truth(num_queries);

  if (num_base == 0 || num_queries == 0 || k == 0) {
    return ground_truth;
  }

  // Validate query_base_indices if provided
  bool use_query_indices = exclude_self && !query_base_indices.empty();
  if (use_query_indices && query_base_indices.size() != num_queries) {
    // Invalid size, fall back to simple q==p exclusion
    use_query_indices = false;
  }

  // Adjust k for exclusion
  size_t actual_k = exclude_self ? k + 1 : k;
  actual_k = std::min(actual_k, num_base);

  // Wrap raw pointers with Eigen::Map (no copy)
  MapConstMatrix base_mat(base_vectors, num_base, dim);
  MapConstMatrix query_mat(query_vectors, num_queries, dim);

  // Handle COSINE by normalizing vectors first
  MatrixXfRowMajor normalized_base;
  MatrixXfRowMajor normalized_queries;
  const MatrixXfRowMajor* base_ptr = nullptr;
  const MatrixXfRowMajor* query_ptr = nullptr;

  if (metric == MetricType::COSINE) {
    // Normalize base vectors
    normalized_base = base_mat;
    VectorXf base_norms = normalized_base.rowwise().norm();
    for (size_t i = 0; i < num_base; ++i) {
      float norm = base_norms(i);
      if (norm < std::numeric_limits<float>::epsilon()) {
        norm = std::numeric_limits<float>::epsilon();
      }
      normalized_base.row(i) /= norm;
    }

    // Normalize query vectors
    normalized_queries = query_mat;
    VectorXf query_norms = normalized_queries.rowwise().norm();
    for (size_t i = 0; i < num_queries; ++i) {
      float norm = query_norms(i);
      if (norm < std::numeric_limits<float>::epsilon()) {
        norm = std::numeric_limits<float>::epsilon();
      }
      normalized_queries.row(i) /= norm;
    }

    base_ptr = &normalized_base;
    query_ptr = &normalized_queries;
    metric = MetricType::L2;  // After normalization, use L2
  } else {
    // Create temporary copies to have consistent pointer types
    // (This is a slight overhead, but simplifies the code significantly)
    normalized_base = base_mat;
    normalized_queries = query_mat;
    base_ptr = &normalized_base;
    query_ptr = &normalized_queries;
  }

  // Compute norms for L2 distance
  VectorXf base_norms_sq;
  VectorXf query_norms_sq;
  if (metric == MetricType::L2) {
    base_norms_sq = base_ptr->rowwise().squaredNorm();
    query_norms_sq = query_ptr->rowwise().squaredNorm();
  }

  // Process queries in batches to limit memory usage
  const size_t batch_size = 512;

  for (size_t batch_start = 0; batch_start < num_queries; batch_start += batch_size) {
    size_t batch_end = std::min(batch_start + batch_size, num_queries);
    size_t current_batch_size = batch_end - batch_start;

    // Compute distance matrix for this batch using Eigen
    // dist_matrix: [current_batch_size x num_base]
    Eigen::MatrixXf dist_matrix(current_batch_size, num_base);

    if (metric == MetricType::L2) {
      // L2 distance: ||a-b||² = ||a||² + ||b||² - 2<a,b>
      // inner_products: [batch_size x num_base] = queries * base^T
      dist_matrix = query_ptr->middleRows(batch_start, current_batch_size) * base_ptr->transpose();
      dist_matrix *= -2.0f;

      // Add squared norms
      for (size_t q = 0; q < current_batch_size; ++q) {
        dist_matrix.row(q).array() += base_norms_sq.array().transpose();
        dist_matrix.row(q).array() += query_norms_sq(batch_start + q);
      }
    } else {
      // IP distance: negative inner product (smaller is better)
      dist_matrix = query_ptr->middleRows(batch_start, current_batch_size) * base_ptr->transpose();
      dist_matrix *= -1.0f;
    }

    // Extract top-k for each query in this batch
#pragma omp parallel for schedule(dynamic, 16)
    for (int64_t q_local = 0; q_local < static_cast<int64_t>(current_batch_size); ++q_local) {
      size_t q = batch_start + q_local;
      MaxHeap heap;

      // Determine which base index to exclude for this query
      size_t exclude_idx = use_query_indices ? query_base_indices[q] : q;

      for (size_t p = 0; p < num_base; ++p) {
        // Skip self if requested (using correct base index mapping)
        if (exclude_self && p == exclude_idx) {
          continue;
        }

        float dist = dist_matrix(q_local, p);

        if (heap.size() < actual_k) {
          heap.emplace(p, dist);
        } else if (dist < heap.top().second) {
          heap.pop();
          heap.emplace(p, dist);
        }
      }

      // Extract results (in reverse order since heap gives max first)
      std::vector<uint64_t>& result = ground_truth[q];
      result.resize(std::min(k, heap.size()));
      for (size_t i = result.size(); i > 0; --i) {
        result[i - 1] = heap.top().first;
        heap.pop();
      }
    }
  }

  return ground_truth;
}

}  // namespace omega
