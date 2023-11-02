#ifndef BF_MLIRGEN_H
#define BF_MLIRGEN_H

#include <memory>

namespace mlir {
class MLIRContext;
template <typename OpTy> class OwningOpRef;
class ModuleOp;
} // namespace mlir

namespace bf {
class ModuleAST;

/// Emit IR for the given BF moduleAST, returns a newly created MLIR module
/// or nullptr on failure.
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context,
                                          ModuleAST &moduleAST);
} // namespace bf

#endif // BF_MLIRGEN_H
