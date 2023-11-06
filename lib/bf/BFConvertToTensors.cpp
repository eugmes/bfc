#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "bf/BFPasses.h"

namespace mlir::bf {
#define GEN_PASS_DEF_BFCONVERTTOTENSORS
#include "bf/BFPasses.h.inc"

namespace {

struct BFTypeConverter : public TypeConverter {
  using TypeConverter::TypeConverter;

  BFTypeConverter() { addConversion(convertType); }

  static LogicalResult convertType(Type t, SmallVectorImpl<Type> &results) {
    if (t.isa<DataIndexType>()) {
      results.push_back(IndexType::get(t.getContext()));
      return success();
    }

    if (t.isa<DataStoreType>()) {
      Type byteType =
          IntegerType::get(t.getContext(), 8); // TODO make configurable
      Type storeType = RankedTensorType::get({30'000}, byteType);
      results.push_back(storeType);
      return success();
    }

    results.push_back(t);
    return success();
  }
};

struct ProgramOpLowering : public OpConversionPattern<ProgramOp> {
public:
  using OpConversionPattern<ProgramOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ProgramOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto *bodyBlock = op.getBody();

    auto &converter = *getTypeConverter();

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
    Type byteType = rewriter.getI8Type();

    Type dataType = converter.convertType(op.getDataArgument().getType());

    auto mainFunc = rewriter.create<func::FuncOp>(loc, "main", funcType);
    auto mainBody = rewriter.createBlock(&mainFunc.getBody());
    Value index = rewriter.create<index::ConstantOp>(loc, 0);
    Value zeroByte = rewriter.create<arith::ConstantIntOp>(loc, 0, byteType);
    Value data = rewriter.create<tensor::SplatOp>(loc, dataType, zeroByte);

    rewriter.mergeBlocks(bodyBlock, mainBody, {index, data});
    rewriter.create<func::ReturnOp>(loc);

    rewriter.eraseOp(op);

    return success();
  }
};

struct ModIndexOpLowering : public OpConversionPattern<ModIndexOp> {
public:
  using OpConversionPattern<ModIndexOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ModIndexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();

    auto amount = op.getAmountAttr().getInt(); // TODO better API
    Value input = adaptor.getInput();

    if (amount == 0) {
      rewriter.replaceOp(op, input);
    } else if (amount < 0) {
      Value c = rewriter.create<index::ConstantOp>(loc, -amount);
      rewriter.replaceOpWithNewOp<index::SubOp>(op, input, c);
    } else {
      Value c = rewriter.create<index::ConstantOp>(loc, amount);
      rewriter.replaceOpWithNewOp<index::AddOp>(op, input, c);
    }

    return success();
  }
};

struct ModDataOpLowering : public OpConversionPattern<ModDataOp> {
public:
  using OpConversionPattern<ModDataOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ModDataOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();

    auto amount = op.getAmountAttr().getInt(); // TODO better API
    Value index = adaptor.getIndex();
    Value data = adaptor.getData();

    if (amount == 0) {
      rewriter.replaceOp(op, data);
    } else {
      Value oldValue =
          rewriter.create<tensor::ExtractOp>(loc, data, ValueRange{index});
      Value newValue;
      if (amount < 0) {
        Value c = rewriter.create<arith::ConstantIntOp>(loc, -amount,
                                                        rewriter.getI8Type());
        newValue = rewriter.create<arith::SubIOp>(loc, oldValue, c);
      } else {
        Value c = rewriter.create<arith::ConstantIntOp>(loc, amount,
                                                        rewriter.getI8Type());
        newValue = rewriter.create<arith::AddIOp>(loc, oldValue, c);
      }
      rewriter.replaceOpWithNewOp<tensor::InsertOp>(op, newValue, data,
                                                    ValueRange{index});
    }

    return success();
  }
};

struct OutputOpLowering : public OpConversionPattern<OutputOp> {
public:
  using OpConversionPattern<OutputOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OutputOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();

    Value index = adaptor.getIndex();
    Value data = adaptor.getData();

    Type intType = rewriter.getI32Type(); // TODO: Make configurable

    Value value =
        rewriter.create<tensor::ExtractOp>(loc, data, ValueRange{index});
    value = rewriter.create<arith::ExtUIOp>(loc, intType, value);
    rewriter.create<func::CallOp>(loc, intType, "putchar", ValueRange{value});
    rewriter.eraseOp(op);

    return success();
  }
};

struct InputOpLowering : public OpConversionPattern<InputOp> {
public:
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
    rewriter.replaceOpWithNewOp<tensor::InsertOp>(op, value, data,
                                                  ValueRange{index});

    return success();
  }
};

struct LoopOpLowering : public OpConversionPattern<LoopOp> {
public:
  using OpConversionPattern<LoopOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LoopOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();

    SmallVector<Type, 2> results;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(), results)))
      return failure();
    SmallVector<Location, 2> locs{rewriter.getUnknownLoc(),
                                  rewriter.getUnknownLoc()};

    auto whileOp = rewriter.replaceOpWithNewOp<scf::WhileOp>(
        op, results, adaptor.getOperands());

    {
      auto &region = whileOp.getRegion(0);
      auto condBlock =
          rewriter.createBlock(&region, region.end(), results, locs);
      Value index = condBlock->getArgument(0);
      Value data = condBlock->getArgument(1);
      Value value =
          rewriter.create<tensor::ExtractOp>(loc, data, ValueRange{index});
      Value zero =
          rewriter.create<arith::ConstantIntOp>(loc, 0, value.getType());
      Value cond = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                                  value, zero);
      rewriter.create<scf::ConditionOp>(loc, cond, ValueRange{index, data});
    }

    {
      auto &region = whileOp.getRegion(1);
      auto bodyBlock =
          rewriter.createBlock(&region, region.end(), results, locs);
      rewriter.mergeBlocks(op.getBody(), bodyBlock, bodyBlock->getArguments());
    }

    return success();
  };
};

struct YieldOpLowering : public OpConversionPattern<YieldOp> {
public:
  using OpConversionPattern<YieldOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, adaptor.getOperands());

    return success();
  }
};

class BFConvertToTensors
    : public impl::BFConvertToTensorsBase<BFConvertToTensors> {
public:
  using impl::BFConvertToTensorsBase<
      BFConvertToTensors>::BFConvertToTensorsBase;

  void runOnOperation() final {
    ConversionTarget target(getContext());

    target.addLegalDialect<scf::SCFDialect, arith::ArithDialect,
                           index::IndexDialect, tensor::TensorDialect,
                           BuiltinDialect, func::FuncDialect>();

    target.addIllegalDialect<BFDialect>();

    BFTypeConverter converter;

    RewritePatternSet patterns(&getContext());
    patterns.add<ProgramOpLowering, ModIndexOpLowering, ModDataOpLowering,
                 InputOpLowering, OutputOpLowering, LoopOpLowering,
                 YieldOpLowering>(converter, patterns.getContext());

    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::bf
