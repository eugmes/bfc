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

#include "bf/BFOps.h"

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
  MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}

  mlir::ModuleOp mlirGen(ModuleAST &moduleAST) {
    auto location = builder.getUnknownLoc();

    theModule = mlir::ModuleOp::create(location);

    builder.setInsertionPointToEnd(theModule.getBody());

    auto program = builder.create<mlir::bf::ProgramOp>(location);

    builder.setInsertionPointToStart(program.getBody());

    // TODO: Add special accessors
    dataIndex = program.getBody()->getArgument(0);
    dataStorage = program.getBody()->getArgument(1);

    mlirGen(*moduleAST.getBody());

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

  mlir::Value dataIndex;
  mlir::Value dataStorage;

  /// Helper conversion for a BF AST location to an MLIR location.
  mlir::Location loc(const Location &loc) {
    return mlir::FileLineColLoc::get(builder.getStringAttr(*loc.file), loc.line,
                                     loc.col);
  }

  void mlirGen(OpASTList &ops) {
    for (auto &op : ops) {
      mlirGen(*op);
    }
  }

  void mlirGen(OpAST &op) {
    switch (op.getKind()) {
    case OpAST::Op_ModPtr:
      mlirGen(llvm::cast<ModPtrOpAST>(op));
      break;
    case OpAST::Op_ModVal:
      mlirGen(llvm::cast<ModValOpAST>(op));
      break;
    case OpAST::Op_Input:
      mlirGen(llvm::cast<InputOpAST>(op));
      break;
    case OpAST::Op_Output:
      mlirGen(llvm::cast<OutputOpAST>(op));
      break;
    case OpAST::Op_Loop:
      mlirGen(llvm::cast<LoopOpAST>(op));
      break;
    }
  }

  void mlirGen(ModPtrOpAST &op) {
    auto location = loc(op.loc());
    dataIndex =
        builder.create<mlir::bf::ModPtrOp>(location, dataIndex, op.getValue());
  }

  void mlirGen(ModValOpAST &op) {
    auto location = loc(op.loc());
    dataStorage = builder.create<mlir::bf::ModDataOp>(
        location, dataIndex, dataStorage, op.getValue());
  }

  void mlirGen(InputOpAST &op) {
    auto location = loc(op.loc());
    dataStorage =
        builder.create<mlir::bf::InputOp>(location, dataIndex, dataStorage);
  }

  void mlirGen(OutputOpAST &op) {
    auto location = loc(op.loc());
    builder.create<mlir::bf::OutputOp>(location, dataIndex, dataStorage);
  }

  void mlirGen(LoopOpAST &op) {
    auto location = loc(op.loc());
    auto loopOp = builder.create<mlir::bf::LoopOp>(location, dataIndex, dataStorage);

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loopOp.getBody());
    // TODO: create accessors
    dataIndex = loopOp.getBody()->getArgument(0);
    dataStorage = loopOp.getBody()->getArgument(1);

    mlirGen(*op.getBody());

    // FIXME: Use location of ] here
    builder.create<mlir::bf::YieldOp>(location, dataIndex, dataStorage);

    // TODO: create accessors
    dataIndex = loopOp.getOutputIndex();
    dataStorage = loopOp.getOutputData();
  }
};

} // namespace

namespace bf {

mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context,
                                          ModuleAST &moduleAST) {
  return MLIRGenImpl(context).mlirGen(moduleAST);
}

} // namespace bf
