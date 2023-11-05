//===- BFOps.cpp - BF dialect ops -------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bf/BFOps.h"
#include "bf/BFDialect.h"

#define GET_OP_CLASSES
#include "bf/BFOps.cpp.inc"

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
  OpAsmParser::Argument indexArgument, dataArgument;
  OpAsmParser::UnresolvedOperand index, data;

  if (parser.parseLParen() || parser.parseArgument(indexArgument) ||
      parser.parseEqual() || parser.parseOperand(index) ||
      parser.parseComma() || parser.parseArgument(dataArgument) ||
      parser.parseEqual() || parser.parseOperand(data) ||
      parser.parseRParen() || parser.parseArrowTypeList(state.types))
    return failure();

  auto &builder = parser.getBuilder();

  if (parser.resolveOperand(index, builder.getType<DataIndexType>(),
                            state.operands) ||
      parser.resolveOperand(data, builder.getType<DataStoreType>(),
                            state.operands))
    return failure();

  indexArgument.type = state.operands[0].getType();
  dataArgument.type = state.operands[1].getType();

  Region *bodyRegion = state.addRegion();
  if (parser.parseRegion(*bodyRegion, {indexArgument, dataArgument}))
    return failure();

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(state.attributes))
    return failure();

  return success();
}

void LoopOp::print(OpAsmPrinter &p) {
  p << "(" << getBody()->getArgument(0) << " = " << getIndex() << ", "
    << getBody()->getArgument(1) << " = " << getData() << ")";
  p.printArrowTypeList(getResultTypes());
  p << " ";
  p.printRegion(getRegion(), false);
  p.printOptionalAttrDict((*this)->getAttrs());
}

mlir::LogicalResult LoopOp::verifyRegions() {
  auto body = getBody();
  if (!body)
    return emitOpError("missing body");

  if (body->getNumArguments() != 2)
    return emitOpError("expected 2 arguments");

  if (!isa<DataIndexType>(getIndexArgument().getType()))
    return emitOpError("first argument should be of '!bf.data_index' type");

  if (!isa<DataStoreType>(getDataArgument().getType()))
    return emitOpError("second argument should be of '!bf.data_store' type");

  // Check that the block is terminated by a YieldOp.
  if (!isa<YieldOp>(body->getTerminator()))
    return emitOpError("body should be terminated with a 'bf.yiedl' op");

  return success();
}

void ProgramOp::build(mlir::OpBuilder &builder, mlir::OperationState &state) {
  Region *bodyRegion = state.addRegion();
  Block *body = new Block();
  bodyRegion->push_back(body);
  body->addArgument(builder.getType<DataIndexType>(), state.location);
  body->addArgument(builder.getType<DataStoreType>(), state.location);
}

mlir::ParseResult ProgramOp::parse(mlir::OpAsmParser &parser,
                                   mlir::OperationState &state) {
  SmallVector<OpAsmParser::Argument, 2> args;
  if (parser.parseArgumentList(args, AsmParser::Delimiter::Paren, true))
    return failure();

  Region *bodyRegion = state.addRegion();
  if (parser.parseRegion(*bodyRegion, args))
    return failure();

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(state.attributes))
    return failure();

  return success();
}

void ProgramOp::print(OpAsmPrinter &p) {
  p << "(";
  llvm::interleaveComma(getBodyRegion().getArguments(), p,
                        [&p](auto it) { p.printRegionArgument(it); });
  p << ") ";
  p.printRegion(getRegion(), false);
  p.printOptionalAttrDict((*this)->getAttrs());
}

mlir::LogicalResult ProgramOp::verifyRegions() {
  auto body = getBody();
  if (!body)
    return emitOpError("missing body");

  if (body->getNumArguments() != 2)
    return emitOpError("expected 2 arguments");

  if (!isa<DataIndexType>(getIndexArgument().getType()))
    return emitOpError("first argument should be of '!bf.data_index' type");

  if (!isa<DataStoreType>(getDataArgument().getType()))
    return emitOpError("second argument should be of '!bf.data_store' type");

  return success();
}