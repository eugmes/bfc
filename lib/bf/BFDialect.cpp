//===- BFDialect.cpp - BF dialect -------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bf/BFDialect.h"
#include "bf/BFOps.h"
#include "bf/BFTypes.h"

using namespace mlir;
using namespace mlir::bf;

#include "bf/BFOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// BF dialect.
//===----------------------------------------------------------------------===//

void BFDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "bf/BFOps.cpp.inc"
      >();
  registerTypes();
}
