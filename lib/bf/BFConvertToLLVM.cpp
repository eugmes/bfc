#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"

#include "bf/BFPasses.h"

namespace mlir::bf {
#define GEN_PASS_DEF_BFCONVERTTOLLVM
#include "bf/BFPasses.h.inc"

namespace {

class BFConvertToLLVM : public impl::BFConvertToLLVMBase<BFConvertToLLVM> {
public:
  using impl::BFConvertToLLVMBase<BFConvertToLLVM>::BFConvertToLLVMBase;

  void runOnOperation() final {
    ConversionTarget target(getContext());

    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalOp<mlir::ModuleOp>();

    auto bufferizePM = PassManager::on<ModuleOp>(&getContext());
    bufferizePM.addPass(bufferization::createOneShotBufferizePass());
    bufferizePM.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
    bufferizePM.addNestedPass<func::FuncOp>(bufferization::createPromoteBuffersToStackPass(1'000'000'000));
    bufferizePM.addPass(memref::createExpandReallocPass());
    bufferizePM.addPass(bufferization::createOwnershipBasedBufferDeallocationPass());
    bufferizePM.addPass(createCanonicalizerPass());
    bufferizePM.addPass(bufferization::createBufferDeallocationSimplificationPass());
    bufferizePM.addPass(bufferization::createLowerDeallocationsPass());
    bufferizePM.addPass(createCSEPass());
    bufferizePM.addPass(createCanonicalizerPass());

    if (failed(runPipeline(bufferizePM, getOperation())))
      return signalPassFailure();

    // During this lowering, we will also be lowering the MemRef types, that are
    // currently being operated on, to a representation in LLVM. To perform this
    // conversion we use a TypeConverter as part of the lowering. This converter
    // details how one type maps to another. This is necessary now that we will
    // be doing more complicated lowerings, involving loop region arguments.
    LLVMTypeConverter typeConverter(&getContext());

    // Now that the conversion target has been defined, we need to provide the
    // patterns used for lowering.
    RewritePatternSet patterns(&getContext());
    index::populateIndexToLLVMConversionPatterns(typeConverter, patterns);
    populateSCFToControlFlowConversionPatterns(patterns);
    arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
    populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);
    populateReconcileUnrealizedCastsPatterns(patterns);
    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::bf
