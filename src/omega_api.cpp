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

void omega_search_reset(OmegaSearchHandle handle) {
  if (!handle || !handle->context) {
    return;
  }

  try {
    handle->context->Reset();
  } catch (...) {
    // Ignore exceptions
  }
}

void omega_search_set_dist_start(OmegaSearchHandle handle, float dist_start) {
  if (!handle || !handle->context) {
    return;
  }

  try {
    handle->context->SetDistStart(dist_start);
  } catch (...) {
    // Ignore exceptions
  }
}

void omega_search_report_visit(OmegaSearchHandle handle, int node_id,
                               float distance, int is_in_topk) {
  if (!handle || !handle->context) {
    return;
  }

  try {
    handle->context->ReportVisit(node_id, distance, is_in_topk != 0);
  } catch (...) {
    // Ignore exceptions
  }
}

void omega_search_report_hop(OmegaSearchHandle handle) {
  if (!handle || !handle->context) {
    return;
  }

  try {
    handle->context->ReportHop();
  } catch (...) {
    // Ignore exceptions
  }
}

int omega_search_should_predict(OmegaSearchHandle handle) {
  if (!handle || !handle->context) {
    return 0;
  }

  try {
    return handle->context->ShouldPredict() ? 1 : 0;
  } catch (...) {
    return 0;
  }
}

int omega_search_should_stop(OmegaSearchHandle handle) {
  if (!handle || !handle->context) {
    return -1;
  }

  try {
    bool should_stop = handle->context->ShouldStopEarly();
    return should_stop ? 1 : 0;
  } catch (...) {
    return -1;
  }
}

void omega_search_get_stats(OmegaSearchHandle handle, int* hops,
                           int* comparisons, int* collected_gt) {
  if (!handle || !handle->context) {
    return;
  }

  try {
    handle->context->GetStats(hops, comparisons, collected_gt);
  } catch (...) {
    // Ignore exceptions
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

void omega_search_enable_training(OmegaSearchHandle handle, int query_id) {
  if (!handle || !handle->context) {
    return;
  }

  try {
    handle->context->EnableTrainingMode(query_id);
  } catch (...) {
    // Ignore exceptions
  }
}

void omega_search_disable_training(OmegaSearchHandle handle) {
  if (!handle || !handle->context) {
    return;
  }

  try {
    handle->context->DisableTrainingMode();
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
