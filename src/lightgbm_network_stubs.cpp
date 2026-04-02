// Copyright 2025-present Yifan Peng
//
// Licensed under the MIT License. See the LICENSE file in the project root for
// license terms.

#ifdef USE_SOCKET
#undef USE_SOCKET
#endif

#include "../lightgbm/src/network/linkers.h"

namespace LightGBM {

namespace {

using RecvIntFn = void (Linkers::*)(int, char*, int) const;
using SendIntFn = void (Linkers::*)(int, char*, int) const;
using SendRecvIntFn = void (Linkers::*)(int, char*, int, int, char*, int);

[[maybe_unused]] RecvIntFn kForceEmitRecvInt =
    static_cast<RecvIntFn>(&Linkers::Recv);
[[maybe_unused]] SendIntFn kForceEmitSendInt =
    static_cast<SendIntFn>(&Linkers::Send);
[[maybe_unused]] SendRecvIntFn kForceEmitSendRecvInt =
    static_cast<SendRecvIntFn>(&Linkers::SendRecv);

}  // namespace

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
