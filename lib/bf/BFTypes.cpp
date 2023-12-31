//===- BFTypes.cpp - BF dialect types ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bf/BFTypes.h"

#include "bf/BFDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::bf;

#define GET_TYPEDEF_CLASSES
#include "bf/BFOpsTypes.cpp.inc"

void BFDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "bf/BFOpsTypes.cpp.inc"
      >();
}
