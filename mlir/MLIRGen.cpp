#include "bf/MLIRGen.h"
#include "bf/AST.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "bf/Lexer.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include <cassert>
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <vector>

using namespace bf;

namespace {
class MLIRGenImpl {
public:
  MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {
    // FIXME: int type width
    intType = mlir::IntegerType::get(&context, 32);
    byteType = mlir::IntegerType::get(&context, 8);
    indexType = mlir::IndexType::get(&context);
  }

  mlir::ModuleOp mlirGen(ModuleAST &moduleAST) {
    auto location = builder.getUnknownLoc();

    theModule = mlir::ModuleOp::create(location);

    auto getcharType =
        mlir::FunctionType::get(builder.getContext(), {}, {intType});
    getchar = mlir::func::FuncOp::create(location, "getchar", getcharType);
    getchar.setVisibility(mlir::SymbolTable::Visibility::Private);
    theModule.push_back(getchar);

    auto putcharType =
        mlir::FunctionType::get(builder.getContext(), {intType}, {intType});
    putchar = mlir::func::FuncOp::create(location, "putchar", putcharType);
    putchar.setVisibility(mlir::SymbolTable::Visibility::Private);
    theModule.push_back(putchar);

    builder.setInsertionPointToEnd(theModule.getBody());

    llvm::SmallVector<mlir::Type, 4> operandTypes;
    auto mainFunction = builder.create<mlir::func::FuncOp>(
        location, "main", builder.getFunctionType(operandTypes, {intType}));
    auto block = mainFunction.addEntryBlock();

    builder.setInsertionPointToEnd(block);

    mlir::Value ptr = builder.create<mlir::index::ConstantOp>(location, 0);

    // FIXME size
    auto dataMemRefType = mlir::MemRefType::get({30'000}, byteType);
    mlir::Value dataMemRef =
        builder.create<mlir::memref::AllocaOp>(location, dataMemRefType);

    mlir::Value loopStart =
        builder.create<mlir::index::ConstantOp>(location, 0);
    mlir::Value loopEnd =
        builder.create<mlir::index::ConstantOp>(location, 30'000); // FIXME
    mlir::Value loopStep = builder.create<mlir::index::ConstantOp>(location, 1);
    auto forOp = builder.create<mlir::scf::ForOp>(location, loopStart, loopEnd,
                                                  loopStep);

    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(forOp.getBody());
      mlir::Value zero =
          builder.create<mlir::arith::ConstantIntOp>(location, 0, byteType);
      mlir::ValueRange idx{forOp.getBody()->getArgument(0)};
      builder.create<mlir::memref::StoreOp>(location, zero, dataMemRef, idx);
    }

    mlirGen(*moduleAST.getBody(), ptr, dataMemRef);

    mlir::Value result =
        builder.create<mlir::arith::ConstantIntOp>(location, 0, intType);

    builder.create<mlir::func::ReturnOp>(location, llvm::ArrayRef(result));

    if (failed(mlir::verify(theModule))) {
      theModule.emitError("module verification error");
      return nullptr;
    }

    return theModule;
  }

private:
  /// A "module" matches a BF source file: containing a list of functions.
  mlir::ModuleOp theModule;

  /// The builder is a helper class to create IR inside a function. The builder
  /// is stateful, in particular it keeps an "insertion point": this is where
  /// the next operations will be introduced.
  mlir::OpBuilder builder;

  mlir::Type intType;
  mlir::Type byteType;
  mlir::Type indexType;

  mlir::func::FuncOp getchar;
  mlir::func::FuncOp putchar;

  /// Helper conversion for a BF AST location to an MLIR location.
  mlir::Location loc(const Location &loc) {
    return mlir::FileLineColLoc::get(builder.getStringAttr(*loc.file), loc.line,
                                     loc.col);
  }

  mlir::Value mlirGen(OpASTList &ops, mlir::Value ptr, mlir::Value data) {
    for (auto &op : ops) {
      ptr = mlirGen(*op, ptr, data);
    }
    return ptr;
  }

  mlir::Value mlirGen(OpAST &op, mlir::Value ptr, mlir::Value data) {
    switch (op.getKind()) {
    case OpAST::Op_ModPtr:
      ptr = mlirGen(llvm::cast<ModPtrOpAST>(op), ptr, data);
      break;
    case OpAST::Op_ModVal:
      ptr = mlirGen(llvm::cast<ModValOpAST>(op), ptr, data);
      break;
    case OpAST::Op_Input:
      ptr = mlirGen(llvm::cast<InputOpAST>(op), ptr, data);
      break;
    case OpAST::Op_Output:
      ptr = mlirGen(llvm::cast<OutputOpAST>(op), ptr, data);
      break;
    case OpAST::Op_Loop:
      ptr = mlirGen(llvm::cast<LoopOpAST>(op), ptr, data);
      break;
    default:
      emitError(loc(op.loc()))
          << "MLIR codegen encountered an unhandled op kind '"
          << llvm::Twine(op.getKind()) << "'";
      return nullptr;
    }
    return ptr;
  }

  mlir::Value mlirGen(ModPtrOpAST &op, mlir::Value ptr, mlir::Value data) {
    auto location = loc(op.loc());

    mlir::Value value =
        builder.create<mlir::index::ConstantOp>(location, op.getValue());
    return builder.create<mlir::arith::AddIOp>(location, ptr, value);
  }

  mlir::Value mlirGen(ModValOpAST &op, mlir::Value ptr, mlir::Value data) {
    auto location = loc(op.loc());
    mlir::ValueRange idx{ptr};

    mlir::Value value = builder.create<mlir::arith::ConstantIntOp>(
        location, op.getValue(), byteType);
    mlir::Value d = builder.create<mlir::memref::LoadOp>(location, data, idx);
    d = builder.create<mlir::arith::AddIOp>(location, d, value);
    builder.create<mlir::memref::StoreOp>(location, d, data, idx);

    return ptr;
  }

  mlir::Value mlirGen(InputOpAST &op, mlir::Value ptr, mlir::Value data) {
    auto location = loc(op.loc());
    mlir::ValueRange idx{ptr};

    mlir::Value d =
        builder.create<mlir::func::CallOp>(location, getchar).getResult(0);
    // FIXME
    d = builder.create<mlir::arith::TruncIOp>(location, byteType, d);
    builder.create<mlir::memref::StoreOp>(location, d, data, idx);

    return ptr;
  }

  mlir::Value mlirGen(OutputOpAST &op, mlir::Value ptr, mlir::Value data) {
    auto location = loc(op.loc());
    mlir::ValueRange idx{ptr};

    mlir::Value d = builder.create<mlir::memref::LoadOp>(location, data, idx);
    // FIXME
    d = builder.create<mlir::arith::ExtUIOp>(location, intType, d);
    builder.create<mlir::func::CallOp>(location, putchar, mlir::ValueRange{d});

    return ptr;
  }

  mlir::Value mlirGen(LoopOpAST &op, mlir::Value ptr, mlir::Value data) {
    auto location = loc(op.loc());
    mlir::TypeRange argTypes{indexType};

    auto whileOp = builder.create<mlir::scf::WhileOp>(location, argTypes,
                                                      mlir::ValueRange{ptr});

    mlir::OpBuilder::InsertionGuard guard(builder);

    auto beforeBlock = builder.createBlock(&whileOp.getBefore());
    // FIXME: why doing this above does not work?
    beforeBlock->addArgument(indexType, builder.getUnknownLoc());
    ptr = beforeBlock->getArgument(0);
    mlir::ValueRange idx{ptr};
    mlir::Value d = builder.create<mlir::memref::LoadOp>(location, data, idx);
    mlir::Value zero =
        builder.create<mlir::arith::ConstantIntOp>(location, 0, byteType);
    mlir::Value condition = builder.create<mlir::arith::CmpIOp>(
        location,
        mlir::arith::CmpIPredicateAttr::get(builder.getContext(),
                                            mlir::arith::CmpIPredicate::ne),
        d, zero);
    builder.create<mlir::scf::ConditionOp>(location, condition,
                                           mlir::ValueRange{ptr});

    auto afterBlock = builder.createBlock(&whileOp.getAfter());
    afterBlock->addArgument(indexType, builder.getUnknownLoc());
    ptr = afterBlock->getArgument(0);
    ptr = mlirGen(*op.getBody(), ptr, data);
    builder.create<mlir::scf::YieldOp>(location, mlir::ValueRange{ptr});

    return whileOp.getResult(0);
  }
};

} // namespace

namespace bf {

mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context,
                                          ModuleAST &moduleAST) {
  return MLIRGenImpl(context).mlirGen(moduleAST);
}

} // namespace bf
