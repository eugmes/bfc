add_mlir_dialect(BFOps bf)
add_mlir_doc(BFDialect BFDialect bf/ -gen-dialect-doc)
add_mlir_doc(BFOps BFOps bf/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS BFPasses.td)
mlir_tablegen(BFPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRBFPassesIncGen)
