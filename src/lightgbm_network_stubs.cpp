// Copyright 2025-present Yifan Peng
//
// Licensed under the MIT License. See the LICENSE file in the project root for
// license terms.

#include "../lightgbm/src/network/linkers.h"

namespace LightGBM {

Linkers::Linkers(Config config) {
  is_init_ = false;
  rank_ = 0;
  num_machines_ = 1;
  if (config.num_machines > 1) {
    Log::Fatal("LightGBM distributed networking support is disabled in omega builds");
  }
}

Linkers::~Linkers() = default;

void Linkers::Recv(int, char*, int) const {
  Log::Fatal("LightGBM distributed networking support is disabled in omega builds");
}

void Linkers::Send(int, char*, int) const {
  Log::Fatal("LightGBM distributed networking support is disabled in omega builds");
}

void Linkers::SendRecv(int, char*, int, int, char*, int) {
  Log::Fatal("LightGBM distributed networking support is disabled in omega builds");
}

}  // namespace LightGBM
