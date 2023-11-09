#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "bf/BFPasses.h"

namespace mlir::bf {
#define GEN_PASS_DEF_BFCONVERTTOMEMREF
#include "bf/BFPasses.h.inc"

namespace {

struct BFTypeConverter : public TypeConverter {
  using TypeConverter::TypeConverter;

  BFTypeConverter() { addConversion(convertType); }

  static LogicalResult convertType(Type t, SmallVectorImpl<Type> &results) {
    if (t.isa<DataStoreType>()) {
      Type byteType =
          IntegerType::get(t.getContext(), 8); // TODO make configurable
      Type storeType = MemRefType::get({30'000}, byteType);
      results.push_back(storeType);
      return success();
    }

    results.push_back(t);
    return success();
  }
};

struct ProgramOpLowering : public OpConversionPattern<ProgramOp> {
  using OpConversionPattern<ProgramOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ProgramOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();

    Type intType = rewriter.getI32Type();
    auto getcharType = rewriter.getFunctionType({}, {intType});
    auto putcharType = rewriter.getFunctionType({intType}, {intType});

    auto symVisAttr = rewriter.getNamedAttr("sym_visibility",
                                            rewriter.getStringAttr("private"));
    rewriter.create<func::FuncOp>(loc, StringRef("getchar"), getcharType,
                                  ArrayRef{symVisAttr});
    rewriter.create<func::FuncOp>(loc, StringRef("putchar"), putcharType,
                                  ArrayRef{symVisAttr});

    auto funcType = rewriter.getFunctionType({}, {});

    auto mainFunc = rewriter.create<func::FuncOp>(loc, "main", funcType);
    auto mainBody = rewriter.createBlock(&mainFunc.getBody());

    rewriter.mergeBlocks(op.getBody(), mainBody);
    rewriter.create<func::ReturnOp>(loc);

    rewriter.eraseOp(op);

    return success();
  }
};

struct AllocOpLowering : public OpConversionPattern<AllocOp> {
  using OpConversionPattern<AllocOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();

    auto &converter = *getTypeConverter();

    Type byteType = rewriter.getI8Type(); // FIXME: make configurable
    Type dataType = converter.convertType(op.getData().getType());

    Value zeroByte = rewriter.create<arith::ConstantIntOp>(loc, 0, byteType);
    Value data = rewriter.replaceOpWithNewOp<memref::AllocaOp>(
        op, cast<MemRefType>(dataType));

    // Initialize the data
    Value low = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value high = rewriter.create<arith::ConstantIndexOp>(loc, 30'000); // FIXME
    Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    auto forOp = rewriter.create<scf::ForOp>(loc, low, high, step);
    auto ip = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(forOp.getBody());
    rewriter.create<memref::StoreOp>(loc, zeroByte, data,
                                     forOp.getBody()->getArguments());
    rewriter.restoreInsertionPoint(ip);

    return success();
  }
};

struct ModDataOpLowering : public OpConversionPattern<ModDataOp> {
  using OpConversionPattern<ModDataOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ModDataOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();

    auto amount = op.getAmountAttr().getInt(); // TODO better API
    Value index = adaptor.getIndex();
    Value data = adaptor.getData();

    Value oldValue =
        rewriter.create<memref::LoadOp>(loc, data, ValueRange{index});

    Value c = rewriter.create<arith::ConstantIntOp>(loc, amount,
                                                    rewriter.getI8Type());
    Value newValue = rewriter.create<arith::AddIOp>(loc, oldValue, c);
    rewriter.replaceOpWithNewOp<memref::StoreOp>(op, newValue, data,
                                                 ValueRange{index});

    return success();
  }
};

struct SetDataOpLowering : public OpConversionPattern<SetDataOp> {
  using OpConversionPattern<SetDataOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SetDataOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();

    auto value = op.getValueAttr().getInt(); // TODO better API
    Value index = adaptor.getIndex();
    Value data = adaptor.getData();

    Value c =
        rewriter.create<arith::ConstantIntOp>(loc, value, rewriter.getI8Type());
    rewriter.replaceOpWithNewOp<memref::StoreOp>(op, c, data,
                                                 ValueRange{index});

    return success();
  }
};

struct OutputOpLowering : public OpConversionPattern<OutputOp> {
  using OpConversionPattern<OutputOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OutputOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();

    Value index = adaptor.getIndex();
    Value data = adaptor.getData();

    Type intType = rewriter.getI32Type(); // TODO: Make configurable

    Value value = rewriter.create<memref::LoadOp>(loc, data, ValueRange{index});
    value = rewriter.create<arith::ExtUIOp>(loc, intType, value);
    rewriter.create<func::CallOp>(loc, intType, "putchar", ValueRange{value});
    rewriter.eraseOp(op);

    return success();
  }
};

struct InputOpLowering : public OpConversionPattern<InputOp> {
  using OpConversionPattern<InputOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(InputOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();

    Value index = adaptor.getIndex();
    Value data = adaptor.getData();

    Type intType = rewriter.getI32Type(); // TODO: Make configurable
    Type byteType = rewriter.getI8Type(); // TODO: Make configurable

    Value value =
        rewriter.create<func::CallOp>(loc, intType, "getchar", ValueRange{})
            .getResult(0);
    value = rewriter.create<arith::TruncIOp>(loc, byteType, value);
    rewriter.replaceOpWithNewOp<memref::StoreOp>(op, value, data,
                                                 ValueRange{index});

    return success();
  }
};

struct LoopOpLowering : public OpConversionPattern<LoopOp> {
  using OpConversionPattern<LoopOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LoopOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();

    Type resultType = getTypeConverter()->convertType(op.getType());

    auto whileOp = rewriter.replaceOpWithNewOp<scf::WhileOp>(
        op, TypeRange{resultType}, ValueRange{adaptor.getIndex()});

    {
      auto &region = whileOp.getRegion(0);
      auto condBlock = rewriter.createBlock(&region, region.end(), {resultType},
                                            {rewriter.getUnknownLoc()});
      Value index = condBlock->getArgument(0);
      Value value = rewriter.create<memref::LoadOp>(loc, adaptor.getData(),
                                                    ValueRange{index});
      Value zero =
          rewriter.create<arith::ConstantIntOp>(loc, 0, value.getType());
      Value cond = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                                  value, zero);
      rewriter.create<scf::ConditionOp>(loc, cond, ValueRange{index});
    }

    {
      auto &region = whileOp.getRegion(1);
      auto bodyBlock = rewriter.createBlock(&region, region.end(), {resultType},
                                            {rewriter.getUnknownLoc()});
      rewriter.mergeBlocks(op.getBody(), bodyBlock, bodyBlock->getArguments());
    }

    return success();
  };
};

struct WhileOpLowering : public OpConversionPattern<WhileOp> {
  using OpConversionPattern<WhileOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(WhileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();

    auto whileOp = rewriter.replaceOpWithNewOp<scf::WhileOp>(op, TypeRange {}, ValueRange {});

    {
      auto &region = whileOp.getBefore();
      rewriter.createBlock(&region);
      Value value = rewriter.create<memref::LoadOp>(loc, adaptor.getData(),
                                                    ValueRange{op.getIndex()});
      Value zero =
          rewriter.create<arith::ConstantIntOp>(loc, 0, value.getType());
      Value cond = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                                  value, zero);
      rewriter.create<scf::ConditionOp>(loc, cond, ValueRange {});
    }

    {
      auto &region = whileOp.getAfter();
      auto bodyBlock = rewriter.createBlock(&region);
      rewriter.mergeBlocks(op.getBody(), bodyBlock, bodyBlock->getArguments());
      rewriter.create<scf::YieldOp>(op.getLoc());
    }

    return success();
  };
};

struct YieldOpLowering : public OpConversionPattern<YieldOp> {
  using OpConversionPattern<YieldOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, adaptor.getOperands());

    return success();
  }
};

struct BFConvertToMemRef
    : public impl::BFConvertToMemRefBase<BFConvertToMemRef> {
  using impl::BFConvertToMemRefBase<BFConvertToMemRef>::BFConvertToMemRefBase;

  void runOnOperation() final {
    ConversionTarget target(getContext());
    target.addLegalOp<ModuleOp>();

    target.addLegalDialect<scf::SCFDialect, arith::ArithDialect,
                           memref::MemRefDialect, func::FuncDialect>();

    target.addIllegalDialect<BFDialect>();

    BFTypeConverter converter;

    RewritePatternSet patterns(&getContext());
    patterns.add<ProgramOpLowering, AllocOpLowering, ModDataOpLowering,
                 SetDataOpLowering, InputOpLowering, OutputOpLowering,
                 LoopOpLowering, WhileOpLowering, YieldOpLowering>(
        converter, patterns.getContext());

    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::bf
