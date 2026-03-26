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

#include "omega/omega_api.h"
#include "omega/model_manager.h"
#include "omega/search_context.h"

// Internal struct definitions (opaque to C API users)
struct OmegaModelManager {
  omega::ModelManager* manager;
};

struct OmegaSearchContext {
  omega::SearchContext* context;
};

// Model management functions

OmegaModelHandle omega_model_create() {
  try {
    OmegaModelHandle handle = new OmegaModelManager();
    handle->manager = new omega::ModelManager();
    return handle;
  } catch (...) {
    return nullptr;
  }
}

int omega_model_load(OmegaModelHandle handle, const char* model_dir) {
  if (!handle || !handle->manager || !model_dir) {
    return -1;
  }

  try {
    bool success = handle->manager->LoadModel(model_dir);
    return success ? 0 : -1;
  } catch (...) {
    return -1;
  }
}

int omega_model_is_loaded(OmegaModelHandle handle) {
  if (!handle || !handle->manager) {
    return 0;
  }

  try {
    return handle->manager->IsLoaded() ? 1 : 0;
  } catch (...) {
    return 0;
  }
}

void omega_model_destroy(OmegaModelHandle handle) {
  if (!handle) {
    return;
  }

  try {
    if (handle->manager) {
      delete handle->manager;
    }
    delete handle;
  } catch (...) {
    // Ignore exceptions during cleanup
  }
}

// Search context functions

OmegaSearchHandle omega_search_create_with_params(OmegaModelHandle model,
                                                   float target_recall,
                                                   int k,
                                                   int window_size) {
  // Allow nullptr model for training mode
  // In training mode, we collect features without making predictions
  const omega::GBDTModel* gbdt_model = nullptr;
  const omega::ModelTables* tables = nullptr;

  if (model && model->manager && model->manager->IsLoaded()) {
    gbdt_model = model->manager->GetModel();
    tables = model->manager->GetTables();
  }

  try {
    OmegaSearchHandle handle = new OmegaSearchContext();
    handle->context = new omega::SearchContext(
        gbdt_model,
        tables,
        target_recall,
        k,
        window_size);
    return handle;
  } catch (...) {
    return nullptr;
  }
}

void omega_search_destroy(OmegaSearchHandle handle) {
  if (!handle) {
    return;
  }

  try {
    if (handle->context) {
      delete handle->context;
    }
    delete handle;
  } catch (...) {
    // Ignore exceptions during cleanup
  }
}

// Phase 5: Training mode functions

void omega_search_enable_training(OmegaSearchHandle handle, int query_id,
                                   const int* ground_truth, size_t gt_count, int k_train) {
  if (!handle || !handle->context) {
    return;
  }

  try {
    std::vector<int> gt_vec;
    if (ground_truth && gt_count > 0) {
      gt_vec.assign(ground_truth, ground_truth + gt_count);
    }
    handle->context->EnableTrainingMode(query_id, gt_vec, k_train);
  } catch (...) {
    // Ignore exceptions
  }
}

const void* omega_search_get_training_records(OmegaSearchHandle handle) {
  if (!handle || !handle->context) {
    return nullptr;
  }

  try {
    const auto& records = handle->context->GetTrainingRecords();
    return static_cast<const void*>(&records);
  } catch (...) {
    return nullptr;
  }
}

size_t omega_search_get_training_records_count(OmegaSearchHandle handle) {
  if (!handle || !handle->context) {
    return 0;
  }

  try {
    return handle->context->GetTrainingRecords().size();
  } catch (...) {
    return 0;
  }
}

const int* omega_search_get_gt_cmps(OmegaSearchHandle handle) {
  if (!handle || !handle->context) {
    return nullptr;
  }

  try {
    const auto& gt_cmps = handle->context->GetGtCmpsPerRank();
    if (gt_cmps.empty()) {
      return nullptr;
    }
    return gt_cmps.data();
  } catch (...) {
    return nullptr;
  }
}

size_t omega_search_get_gt_cmps_count(OmegaSearchHandle handle) {
  if (!handle || !handle->context) {
    return 0;
  }

  try {
    return handle->context->GetGtCmpsPerRank().size();
  } catch (...) {
    return 0;
  }
}

int omega_search_get_total_cmps(OmegaSearchHandle handle) {
  if (!handle || !handle->context) {
    return 0;
  }

  try {
    return handle->context->GetTotalCmps();
  } catch (...) {
    return 0;
  }
}

omega::SearchContext* omega_search_get_cpp_context(OmegaSearchHandle handle) {
  if (!handle) {
    return nullptr;
  }
  return handle->context;
}
