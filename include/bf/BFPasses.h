//===- BFPasses.h - BF passes  ----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef BF_BFPASSES_H
#define BF_BFPASSES_H

#include "bf/BFDialect.h"
#include "bf/BFOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace bf {
#define GEN_PASS_DECL
#include "bf/BFPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "bf/BFPasses.h.inc"

} // namespace bf
} // namespace mlir

#endif // BF_BFPASSES_H
