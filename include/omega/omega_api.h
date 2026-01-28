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

// Create a new search context
// Parameters:
//   model: Model manager handle (must be loaded)
// Returns: Handle to search context, or NULL on failure
OmegaSearchHandle omega_search_create(OmegaModelHandle model);

// Reset search context for a new query
// Parameters:
//   handle: Search context handle
void omega_search_reset(OmegaSearchHandle handle);

// Update search state with new information
// Parameters:
//   handle: Search context handle
//   hops: Current hop count
//   comparisons: Current comparison count
//   distance: Latest distance value
void omega_search_update(OmegaSearchHandle handle, int hops,
                        int comparisons, float distance);

// Set the distance to the first result
// Parameters:
//   handle: Search context handle
//   dist_1st: Distance to first result
void omega_search_set_dist_1st(OmegaSearchHandle handle, float dist_1st);

// Set the distance to the start node
// Parameters:
//   handle: Search context handle
//   dist_start: Distance to start node
void omega_search_set_dist_start(OmegaSearchHandle handle, float dist_start);

// Check if search should stop early
// Parameters:
//   handle: Search context handle
//   target_recall: Target recall value (0.0 to 1.0)
// Returns: 1 if should stop, 0 if should continue, -1 on error
int omega_search_should_stop(OmegaSearchHandle handle, float target_recall);

// Get optimal EF value for current search state
// Parameters:
//   handle: Search context handle
//   target_recall: Target recall value (0.0 to 1.0)
//   current_ef: Current EF value
// Returns: Optimal EF value, or current_ef on error
int omega_search_get_optimal_ef(OmegaSearchHandle handle,
                               float target_recall, int current_ef);

// Destroy search context and free resources
// Parameters:
//   handle: Search context handle
void omega_search_destroy(OmegaSearchHandle handle);

#ifdef __cplusplus
}
#endif

#endif  // ZVEC_THIRDPARTY_OMEGA_INCLUDE_OMEGA_API_H_
