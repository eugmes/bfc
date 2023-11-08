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

def BFConvertToMemRef: Pass<"convert-bf-to-memref", "ModuleOp"> {
    let summary = "Convert BF dialect into memref and other standard MLIR dialects.";
    let dependentDialects = [
        "memref::MemRefDialect",
        "scf::SCFDialect",
        "arith::ArithDialect",
        "func::FuncDialect",
    ];
}

def BFConvertToLLVM: Pass<"convert-bf-to-llvm", "ModuleOp"> {
    let dependentDialects = [
        "LLVM::LLVMDialect",
        "cf::ControlFlowDialect",
    ];
}

#endif
