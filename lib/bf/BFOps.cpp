//===- BFOps.cpp - BF dialect ops -------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bf/BFOps.h"
#include "bf/BFDialect.h"

#include "mlir/Support/LogicalResult.h"

#define GET_OP_CLASSES
#include "bf/BFOps.cpp.inc"

using namespace mlir;
using namespace llvm;
using namespace mlir::bf;

void ModDataOp::build(OpBuilder &builder, OperationState &state, Value index,
                      Value data, int64_t amount) {
  state.addOperands({index, data});
  state.addAttribute(getAmountAttrName(state.name),
                     builder.getIndexAttr(amount));
}

namespace {
struct CombineModData : public OpRewritePattern<ModDataOp> {
  using OpRewritePattern<ModDataOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ModDataOp op,
                                PatternRewriter &rewriter) const final {
    // TODO allow to combine ops that are not immediately following each other
    auto prev = op->getPrevNode();
    if (!prev)
      return failure();

    auto prevOp = dyn_cast<ModDataOp>(op->getPrevNode());
    if (!prevOp || (prevOp.getIndex() != op.getIndex()) ||
        (prevOp.getData() != op.getData()))
      return failure();

    auto newAmount =
        prevOp.getAmountAttr().getInt() + op.getAmountAttr().getInt();

    rewriter.eraseOp(prevOp);
    if (newAmount != 0) {
      rewriter.startRootUpdate(op);
      op.setAmount(newAmount);
      rewriter.finalizeRootUpdate(op);
    } else {
      rewriter.eraseOp(op);
    }

    return success();
  }
};

struct CombineSetDataAndModData : public OpRewritePattern<ModDataOp> {
  using OpRewritePattern<ModDataOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ModDataOp op,
                                PatternRewriter &rewriter) const final {
    // TODO allow to combine ops that are not immediately following each other
    auto prev = op->getPrevNode();
    if (!prev)
      return failure();

    auto prevOp = dyn_cast<SetDataOp>(op->getPrevNode());
    if (!prevOp || (prevOp.getIndex() != op.getIndex()) ||
        (prevOp.getData() != op.getData()))
      return failure();

    auto newValue =
        prevOp.getValueAttr().getInt() + op.getAmountAttr().getInt();

    rewriter.eraseOp(prevOp);
    rewriter.replaceOpWithNewOp<SetDataOp>(op, op.getIndex(), op.getData(),
                                           newValue);

    return success();
  }
};
} // namespace

void ModDataOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<CombineModData, CombineSetDataAndModData>(context);
}

void SetDataOp::build(OpBuilder &builder, OperationState &state, Value index,
                      Value data, int64_t value) {
  state.addOperands({index, data});
  state.addAttribute(getValueAttrName(state.name), builder.getIndexAttr(value));
}

void LoopOp::build(OpBuilder &builder, OperationState &state, Value index,
                   Value data) {
  state.addOperands({index, data});
  state.addTypes({index.getType()});
  Region *bodyRegion = state.addRegion();
  Block *body = new Block();
  bodyRegion->push_back(body);
  body->addArgument(index.getType(), state.location);
}

ParseResult LoopOp::parse(OpAsmParser &parser, OperationState &state) {
  OpAsmParser::Argument indexArgument;
  OpAsmParser::UnresolvedOperand index, data;

  if (parser.parseLParen() || parser.parseArgument(indexArgument) ||
      parser.parseEqual() || parser.parseOperand(index) ||
      parser.parseComma() || parser.parseOperand(data) ||
      parser.parseRParen() || parser.parseArrowTypeList(state.types))
    return failure();

  auto &builder = parser.getBuilder();

  if (parser.resolveOperand(index, builder.getIndexType(), state.operands) ||
      parser.resolveOperand(data, builder.getType<DataStoreType>(),
                            state.operands))
    return failure();

  indexArgument.type = state.operands[0].getType();

  Region *bodyRegion = state.addRegion();
  if (parser.parseRegion(*bodyRegion, {indexArgument}))
    return failure();

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(state.attributes))
    return failure();

  return success();
}

void LoopOp::print(OpAsmPrinter &p) {
  p << "(" << getBody()->getArgument(0) << " = " << getIndex() << ", "
    << getData() << ")";
  p.printArrowTypeList(TypeRange{getType()});
  p << " ";
  p.printRegion(getRegion(), false);
  p.printOptionalAttrDict((*this)->getAttrs());
}

LogicalResult LoopOp::verifyRegions() {
  auto body = getBody();
  if (!body)
    return emitOpError("missing body");

  if (body->getNumArguments() != 1)
    return emitOpError("expected 1 argument");

  if (!getIndexArgument().getType().isIndex())
    return emitOpError("first argument should be of 'index' type");

  // Check that the block is terminated by a YieldOp.
  if (!isa<YieldOp>(body->getTerminator()))
    return emitOpError("body should be terminated with a 'bf.yiedl' op");

  return success();
}

namespace {
struct SmplifyLoop : public OpRewritePattern<LoopOp> {
  using OpRewritePattern<LoopOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LoopOp op,
                                PatternRewriter &rewriter) const final {
    auto body = op.getBody();

    auto yieldOp = cast<YieldOp>(body->getTerminator());
    if ((yieldOp.getIndex() != op.getIndexArgument()) &&
        (yieldOp.getIndex() != op.getIndex()))
      return failure();

    rewriter.eraseOp(body->getTerminator());

    auto whileOp =
        rewriter.create<WhileOp>(op.getLoc(), op.getIndex(), op.getData());
    rewriter.mergeBlocks(body, whileOp.getBody(), {op.getIndex()});

    rewriter.replaceOp(op, {op.getIndex()});

    return success();
  }
};
} // namespace

void LoopOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<SmplifyLoop>(context);
}

void WhileOp::build(OpBuilder &builder, OperationState &state, Value index,
                    Value data) {
  state.addOperands({index, data});
  Region *bodyRegion = state.addRegion();
  Block *body = new Block();
  bodyRegion->push_back(body);
}

namespace {
struct ReplaceSetToZeroLoop : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const final {
    auto body = op.getBody();
    if (std::size(body->getOperations()) != 1)
      return failure();

    auto modDataOp = dyn_cast<ModDataOp>(body->front());
    if (!modDataOp)
      return failure();
    if (abs(modDataOp.getAmountAttr().getInt()) != 1)
      return failure();
    if (modDataOp.getOperands() != op.getOperands())
      return failure();

    rewriter.replaceOpWithNewOp<SetDataOp>(op, op.getIndex(), op.getData(), 0);

    return success();
  }
};
} // namespace

void WhileOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.add<ReplaceSetToZeroLoop>(context);
}

void ProgramOp::build(OpBuilder &builder, OperationState &state) {
  Region *bodyRegion = state.addRegion();
  Block *body = new Block();
  bodyRegion->push_back(body);
}
