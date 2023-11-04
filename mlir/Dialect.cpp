#include "bf/BFOps.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <string>

#include <mlir/Dialect/Index/IR/IndexDialect.h>

using namespace mlir::bf;

void ModPtrOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                     mlir::Value input, int64_t amount) {
  state.addOperands({input});
  state.addTypes({input.getType()});
  state.addAttribute(getAmountAttrName(state.name),
                     builder.getIndexAttr(amount));
}

void ModDataOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                      mlir::Value index, mlir::Value data, int64_t amount) {
  state.addOperands({index, data});
  state.addTypes({data.getType()});
  state.addAttribute(getAmountAttrName(state.name),
                     builder.getIndexAttr(amount));
}

void LoopOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   mlir::Value index, mlir::Value data) {
  state.addOperands({index, data});
  state.addTypes({index.getType(), data.getType()});
  Region *bodyRegion = state.addRegion();
  Block *body = new Block();
  bodyRegion->push_back(body);
  body->addArgument(index.getType(), state.location);
  body->addArgument(data.getType(), state.location);
}

mlir::ParseResult LoopOp::parse(mlir::OpAsmParser &parser,
                                mlir::OperationState &state) {
  // TODO
}

void LoopOp::print(OpAsmPrinter &p) {
  p << "(" << getBody()->getArgument(0) << " = " << getOperand(0) << ", "
    << getBody()->getArgument(1) << " = " << getOperand(1) << ") ";
  p.printRegion(getRegion(), false);
  p.printOptionalAttrDict((*this)->getAttrs());
}

void ProgramOp::build(mlir::OpBuilder &builder, mlir::OperationState &state) {
  Region *bodyRegion = state.addRegion();
  Block *body = new Block();
  bodyRegion->push_back(body);
  body->addArgument(DataIndexType::get(builder.getContext()), state.location);
  body->addArgument(DataStoreType::get(builder.getContext()), state.location);
}

mlir::ParseResult ProgramOp::parse(mlir::OpAsmParser &parser,
                                   mlir::OperationState &state) {
  // TODO
}

void ProgramOp::print(OpAsmPrinter &p) {
  p << "(" << getBody()->getArguments() << ") ";
  p.printRegion(getRegion(), false);
  p.printOptionalAttrDict((*this)->getAttrs());
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#include "bf/BFOpsDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "bf/BFOpsTypes.cpp.inc"

#define GET_OP_CLASSES
#include "bf/BFOps.cpp.inc"

void BFDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "bf/BFOpsTypes.cpp.inc"
      >();
}

void BFDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "bf/BFOps.cpp.inc"
      >();
  registerTypes();
}
