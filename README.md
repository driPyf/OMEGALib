# OMEGA - One-Model Efficient Generalized ANN

OMEGA is a lightweight adaptive search library for Approximate Nearest Neighbor (ANN) search. It provides dynamic early stopping using decision tree models to reduce distance computations while maintaining target recall.

## Features

- **Lightweight Decision Tree Inference**: Minimal GBDT implementation (~2-3 MB)
- **No Heavy Dependencies**: Extracted tree training/inference without LightGBM library
- **Cross-Platform**: Supports Linux, macOS, Windows
- **Well-Tested**: Comprehensive unit tests with Google Test
- **Google C++ Style**: Follows Google C++ coding standards

## Directory Structure

```
omega/
├── include/           # Public headers
│   └── tree_inference.h
├── src/              # Implementation
│   └── tree_inference.cpp
├── tests/            # Unit tests
│   └── tree_inference_test.cpp
├── python/           # Python bindings (future)
└── CMakeLists.txt    # Build configuration
```

## Building

```bash
mkdir build && cd build
cmake .. -DOMEGA_BUILD_TESTS=ON
make -j8
```

## Running Tests

```bash
./tree_inference_test
```

## Usage Example

```cpp
#include "tree_inference.h"

// Load a trained GBDT model
omega::GBDTModel model;
model.LoadFromFile("model.txt");

// Predict probability
double features[11] = {1.0, 2.0, ..., 11.0};
double probability = model.Predict(features, 11);
```

## Integration with zvec

This library is designed to be integrated into Alibaba's zvec vector database as a submodule:

```bash
cd zvec/thirdparty
git submodule add <omega-repo-url> omega
```

## License

Apache License 2.0 - See LICENSE file for details

## Contributing

This is part of the OMEGA integration project for zvec. For questions or contributions, please refer to the main zvec repository.
