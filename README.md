# OMEGALib

OMEGALib is a C++ library for OMEGA-based adaptive early termination in approximate nearest neighbor (ANN) search.

Its main goal is to make OMEGA reusable as a standalone algorithm library rather than tying the implementation to a single vector database. zvec is currently an important integration target, but OMEGALib is intended to remain generally usable by other ANN systems that want:

- query-time adaptive early termination
- OMEGA model training in C++
- model/table loading and runtime management
- a lightweight runtime API for integrating OMEGA with an existing search loop

## What OMEGALib Provides

OMEGALib currently includes four main parts:

- **SearchContext**
  Maintains query-time state for OMEGA, including traversal-window statistics, target recall handling, and stop/continue decisions.

- **ModelManager**
  Loads and manages OMEGA model artifacts such as `model.txt`, `threshold_table.txt`, `interval_table.txt`, `gt_collected_table.txt`, and `gt_cmps_all_table.txt`.

- **OmegaTrainer**
  Trains OMEGA models in C++ using the LightGBM C API and generates the runtime tables required by `SearchContext`.

- **C API / integration layer**
  Exposes runtime handles for model loading, search-context creation, and training-data extraction so external systems can integrate OMEGA into their own ANN search loops.

## Repository Layout

```text
OMEGALib/
├── include/omega/
│   ├── ground_truth.h
│   ├── model_manager.h
│   ├── omega_api.h
│   ├── omega_trainer.h
│   ├── search_context.h
│   └── tree_inference.h
├── src/
│   ├── ground_truth.cpp
│   ├── model_manager.cpp
│   ├── omega_api.cpp
│   ├── omega_trainer.cpp
│   ├── search_context.cpp
│   └── tree_inference.cpp
├── tests/
│   ├── model_manager_test.cpp
│   └── tree_inference_test.cpp
├── eigen/        # header-only Eigen dependency
├── lightgbm/     # vendored LightGBM dependency
└── CMakeLists.txt
```

## Dependencies

OMEGALib currently depends on:

- **LightGBM**
  Used for training OMEGA models through the LightGBM C API.

- **OpenMP**
  Required by the current LightGBM-based build.

- **Eigen**
  Used for fast ground-truth computation utilities.

This means OMEGALib is not just a minimal tree-inference library anymore. It is a full training + runtime library for OMEGA.

## Building

```bash
mkdir build
cd build
cmake .. -DBUILD_TESTING=ON
cmake --build . -j
```

## Running Tests

If tests are enabled, OMEGALib currently provides GoogleTest-based unit tests such as:

- `omega_model_manager_test`

Depending on the build configuration, tests can be run via `ctest`:

```bash
ctest --output-on-failure
```

## Public Interfaces

### Runtime API

The main runtime entry points are exposed in:

- `include/omega/omega_api.h`

This API supports:

- loading model artifacts from a directory
- creating OMEGA search contexts with `target_recall`, `k`, and `window_size`
- enabling training mode and extracting collected training data

### Training API

The training entry point is exposed in:

- `include/omega/omega_trainer.h`

This API supports:

- training OMEGA models from collected training records
- generating the threshold / interval / ground-truth tables needed at runtime

## Typical Integration Pattern

OMEGALib is designed to sit on top of an existing ANN search loop rather than replace it.

A typical integration looks like this:

1. The host ANN system runs its normal search algorithm.
2. During search, it reports query-state updates into OMEGA's `SearchContext`.
3. OMEGA decides whether the target recall is likely to have been reached.
4. The host system either stops early or continues searching.

This makes OMEGALib especially suitable for systems that already have a mature ANN implementation and want to add adaptive early termination without maintaining a second search engine.

## Integration with zvec

zvec is currently one of the main production-oriented integrations of OMEGALib.

In zvec:

- HNSW remains the underlying search engine
- OMEGALib provides query-time stopping logic and offline model training
- model artifacts are persisted under `omega_model/`
- runtime behavior falls back to plain HNSW when no OMEGA model is available or the dataset is below threshold

OMEGALib itself is not specific to zvec, and the library is intended to stay reusable by other ANN systems.

## Status

OMEGALib is under active development. The current codebase already supports:

- C++ model training
- model/table loading
- adaptive early-termination runtime state
- unit tests for key runtime components

Future work may include:

- cleaner standalone packaging
- broader test coverage
- additional integration examples outside zvec

## License

Please refer to the repository license files and the vendored third-party dependencies for current licensing details.
