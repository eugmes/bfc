#ifndef BF_PASSES_H
#define BF_PASSES_H

#include <memory>

namespace mlir {

class Pass;

namespace bf {

std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

} // namespace bf

} // namespace mlir

#endif // BF_PASSES_H
