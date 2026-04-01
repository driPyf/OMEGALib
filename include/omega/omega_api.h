// Copyright 2025-present Yifan Peng
//
// Licensed under the MIT License. See the LICENSE file in the project root for
// license terms.

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

// Destroy search context and free resources
// Parameters:
//   handle: Search context handle
void omega_search_destroy(OmegaSearchHandle handle);

// Training-mode functions

// Enable training mode for collecting features with ground truth for real-time label computation
// Parameters:
//   handle: Search context handle
//   query_id: ID of the current query
//   ground_truth: Array of ground truth node IDs (sorted by rank)
//   gt_count: Number of ground truth nodes
//   k_train: Number of GT nodes to check for label (typically 1)
void omega_search_enable_training(OmegaSearchHandle handle, int query_id,
                                   const int* ground_truth, size_t gt_count, int k_train);

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
