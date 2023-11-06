//===- BFPasses.td - BF dialect passes ---------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef BF_PASSES
#define BF_PASSES

include "mlir/Pass/PassBase.td"

def BFConvertToTensors: Pass<"convert-bf-to-tensors", "ModuleOp"> {
    let summary = "Convert BF dialect into Tensor and other standard MLIR dialects.";
    let dependentDialects = [
        "tensor::TensorDialect",
        "scf::SCFDialect",
        "arith::ArithDialect",
        "func::FuncDialect",
        "index::IndexDialect",
    ];
}

#endif
