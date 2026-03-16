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

#ifdef __cplusplus
}
#endif

#endif  // ZVEC_THIRDPARTY_OMEGA_INCLUDE_OMEGA_API_H_
