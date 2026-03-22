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

#ifndef ZVEC_THIRDPARTY_OMEGA_INCLUDE_OMEGA_API_H_
#define ZVEC_THIRDPARTY_OMEGA_INCLUDE_OMEGA_API_H_

#include <cstddef>  // For size_t

#ifdef __cplusplus
namespace omega {
class SearchContext;
}
extern "C" {
#endif

// Opaque handles for C API
typedef struct OmegaModelManager* OmegaModelHandle;
typedef struct OmegaSearchContext* OmegaSearchHandle;

// Model management functions

// Create a new model manager
// Returns: Handle to model manager, or NULL on failure
OmegaModelHandle omega_model_create();

// Load model and tables from directory
// Parameters:
//   handle: Model manager handle
//   model_dir: Path to directory containing model files
// Returns: 0 on success, non-zero on failure
int omega_model_load(OmegaModelHandle handle, const char* model_dir);

// Check if model is loaded
// Parameters:
//   handle: Model manager handle
// Returns: 1 if loaded, 0 otherwise
int omega_model_is_loaded(OmegaModelHandle handle);

// Destroy model manager and free resources
// Parameters:
//   handle: Model manager handle
void omega_model_destroy(OmegaModelHandle handle);

// Search context functions

// Create a new search context with parameters (stateful interface)
// Parameters:
//   model: Model manager handle (must be loaded)
//   target_recall: Target recall value (0.0 to 1.0)
//   k: Top-K value
//   window_size: Traversal window size (default 100)
// Returns: Handle to search context, or NULL on failure
OmegaSearchHandle omega_search_create_with_params(OmegaModelHandle model,
                                                   float target_recall,
                                                   int k,
                                                   int window_size);

// Reset search context for a new query
// Parameters:
//   handle: Search context handle
void omega_search_reset(OmegaSearchHandle handle);

// Set the distance to the start node (called once at search start)
// Parameters:
//   handle: Search context handle
//   dist_start: Distance to start node
void omega_search_set_dist_start(OmegaSearchHandle handle, float dist_start);

// Report a node visit during search (stateful interface)
// Parameters:
//   handle: Search context handle
//   node_id: ID of the visited node
//   distance: Distance to the visited node
//   is_in_topk: 1 if node is in current top-K results, 0 otherwise
void omega_search_report_visit(OmegaSearchHandle handle, int node_id,
                               float distance, int is_in_topk);

// Report a node visit and let OMEGA maintain the result-set top-k internally.
// Parameters:
//   handle: Search context handle
//   node_id: ID of the visited node
//   distance: Distance to the visited node
//   should_consider: 1 if the HNSW ef-heap logic says this candidate should be
//                    considered for insertion, 0 otherwise
// Returns:
//   1 if inserted into the current result-set top-k, 0 otherwise
int omega_search_report_visit_candidate(OmegaSearchHandle handle, int node_id,
                                        float distance, int should_consider);

// Report a hop during search (stateful interface)
// Parameters:
//   handle: Search context handle
void omega_search_report_hop(OmegaSearchHandle handle);

// Check if should perform prediction now (stateful interface)
// Based on interval_table and current comparison count
// Parameters:
//   handle: Search context handle
// Returns: 1 if should predict, 0 otherwise
int omega_search_should_predict(OmegaSearchHandle handle);

// Check if search should stop early
// Parameters:
//   handle: Search context handle
// Returns: 1 if should stop, 0 if should continue, -1 on error
int omega_search_should_stop(OmegaSearchHandle handle);

// Get current search statistics (stateful interface)
// Parameters:
//   handle: Search context handle
//   hops: Output pointer for hop count (can be NULL)
//   comparisons: Output pointer for comparison count (can be NULL)
//   collected_gt: Output pointer for collected ground truth count (can be NULL)
void omega_search_get_stats(OmegaSearchHandle handle, int* hops,
                           int* comparisons, int* collected_gt);

// Get OMEGA debug statistics for the current search state.
// Parameters:
//   handle: Search context handle
//   predicted_recall_avg: Output pointer for latest predicted recall average
//   predicted_recall_at_target: Output pointer for latest rank-target prediction
//   early_stop_hit: Output pointer for whether early stop has triggered
void omega_search_get_debug_stats(OmegaSearchHandle handle,
                                  float* predicted_recall_avg,
                                  float* predicted_recall_at_target,
                                  int* early_stop_hit,
                                  unsigned long long* should_stop_calls,
                                  unsigned long long* prediction_calls,
                                  unsigned long long* should_stop_time_ns,
                                  unsigned long long* prediction_eval_time_ns,
                                  unsigned long long* sorted_window_time_ns,
                                  unsigned long long* average_recall_eval_time_ns,
                                  unsigned long long* prediction_feature_prep_time_ns,
                                  unsigned long long* collected_gt_advance_count,
                                  unsigned long long* should_stop_calls_with_advance,
                                  unsigned long long* max_prediction_calls_per_should_stop);

// Destroy search context and free resources
// Parameters:
//   handle: Search context handle
void omega_search_destroy(OmegaSearchHandle handle);

// Training mode functions (Phase 5)

// Enable training mode for collecting features with ground truth for real-time label computation
// Parameters:
//   handle: Search context handle
//   query_id: ID of the current query
//   ground_truth: Array of ground truth node IDs (sorted by rank)
//   gt_count: Number of ground truth nodes
//   k_train: Number of GT nodes to check for label (typically 1)
void omega_search_enable_training(OmegaSearchHandle handle, int query_id,
                                   const int* ground_truth, size_t gt_count, int k_train);

// Disable training mode
// Parameters:
//   handle: Search context handle
void omega_search_disable_training(OmegaSearchHandle handle);

// Get training records collected so far
// Parameters:
//   handle: Search context handle
// Returns: Pointer to training records array (opaque)
const void* omega_search_get_training_records(OmegaSearchHandle handle);

// Get number of training records collected
// Parameters:
//   handle: Search context handle
// Returns: Number of training records
size_t omega_search_get_training_records_count(OmegaSearchHandle handle);

// Get gt_cmps data for this query (cmps value when each GT rank was found)
// Parameters:
//   handle: Search context handle
// Returns: Pointer to array of int, size = ground_truth count
//          Value -1 means GT rank was not found (use total_cmps instead)
const int* omega_search_get_gt_cmps(OmegaSearchHandle handle);

// Get count of gt_cmps array
// Parameters:
//   handle: Search context handle
// Returns: Number of elements in gt_cmps array (same as ground_truth count)
size_t omega_search_get_gt_cmps_count(OmegaSearchHandle handle);

// Get total cmps for this search (useful for unfound GT ranks)
// Parameters:
//   handle: Search context handle
// Returns: Total comparison count
int omega_search_get_total_cmps(OmegaSearchHandle handle);

#ifdef __cplusplus
}

// C++ fast path for hot call sites inside zvec. This bypasses the C wrapper
// overhead while keeping the public C API unchanged for other users.
omega::SearchContext* omega_search_get_cpp_context(OmegaSearchHandle handle);
#endif

#endif  // ZVEC_THIRDPARTY_OMEGA_INCLUDE_OMEGA_API_H_
