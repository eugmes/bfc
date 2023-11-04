#ifndef BF_BFOPS_H
#define BF_BFOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

#include "bf/BFOpsDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "bf/BFOpsTypes.h.inc"

#define GET_OP_CLASSES
#include "bf/BFOps.h.inc"

#endif
