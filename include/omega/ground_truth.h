// Copyright 2025-present Yifan Peng
//
// Licensed under the MIT License. See the LICENSE file in the project root for
// license terms.

#ifndef OMEGA_GROUND_TRUTH_H_
#define OMEGA_GROUND_TRUTH_H_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace omega {

/**
 * @brief Metric type for distance computation
 */
enum class MetricType {
  L2,           // Euclidean distance (L2)
  IP,           // Inner product (negative, so smaller is better)
  COSINE        // Cosine similarity (converted to L2 after normalization)
};

/**
 * @brief Compute exact k-NN ground truth using fast batch matrix multiplication.
 *
 * This function uses Eigen for efficient distance computation (cross-platform).
 * The algorithm:
 * 1. For L2: ||a-b||² = ||a||² + ||b||² - 2<a,b>, uses matrix multiply for batch inner product
 * 2. For IP: directly uses matrix multiply for batch inner product
 * 3. For COSINE: normalizes vectors first, then uses L2
 *
 * @param base_vectors     Base vectors, row-major [num_base x dim]
 * @param query_vectors    Query vectors, row-major [num_queries x dim]
 * @param num_base         Number of base vectors
 * @param num_queries      Number of query vectors
 * @param dim              Vector dimension
 * @param k                Number of nearest neighbors to find
 * @param metric           Distance metric to use
 * @param exclude_self     If true, exclude each query's corresponding base vector from results
 *                         (useful for held-out evaluation where queries are sampled from base)
 * @param query_base_indices If exclude_self is true and this is non-empty, query q will
 *                         exclude base vector query_base_indices[q] (size must be num_queries).
 *                         If empty and exclude_self is true, query q excludes base vector q.
 * @return                 Ground truth indices, [num_queries x k], each row contains
 *                         k nearest neighbor indices sorted by distance
 */
std::vector<std::vector<uint64_t>> ComputeGroundTruth(
    const float* base_vectors,
    const float* query_vectors,
    size_t num_base,
    size_t num_queries,
    size_t dim,
    size_t k,
    MetricType metric = MetricType::IP,
    bool exclude_self = false,
    const std::vector<uint64_t>& query_base_indices = {});

}  // namespace omega

#endif  // OMEGA_GROUND_TRUTH_H_
