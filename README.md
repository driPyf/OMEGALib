# OMEGALib

OMEGALib is a C++ library for OMEGA-based adaptive early termination in approximate nearest neighbor search.

It is designed for systems that already have an ANN search loop and want to add:

- query-time adaptive stopping
- C++ model training
- model and table loading
- a small runtime API for stop/continue decisions

## Build

**Requirements**: CMake, C++17, OpenMP

```bash
mkdir build
cd build
cmake .. -DBUILD_TESTING=ON
cmake --build . -j
```

## Test

```bash
ctest --output-on-failure
```

## Main Interfaces

- `include/omega/search_context.h`
  Maintains per-query runtime state and adaptive stop decisions.
- `include/omega/model_manager.h`
  Loads the trained model and auxiliary tables.
- `include/omega/omega_trainer.h`
  Trains OMEGA models and generates runtime artifacts.
- `include/omega/omega_api.h`
  Exposes a lightweight C API for integration.

## Typical Usage

OMEGALib does not replace the underlying ANN algorithm. The host system keeps its normal search loop, reports query progress into OMEGA, and stops when OMEGA predicts that the target recall has been reached.

## Reference

```bibtex
@misc{peng2026efficientvectorsearchwild,
      title={Efficient Vector Search in the Wild: One Model for Multi-K Queries},
      author={Yifan Peng and Jiafei Fan and Xingda Wei and Sijie Shen and Rong Chen and Jianning Wang and Xiaojian Luo and Wenyuan Yu and Jingren Zhou and Haibo Chen},
      year={2026},
      eprint={2603.06159},
      archivePrefix={arXiv},
      primaryClass={cs.DB},
      url={https://arxiv.org/abs/2603.06159},
}
```

## License

This repository is released under the MIT License. Vendored third-party dependencies keep their own original licenses.
