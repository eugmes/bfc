get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    ${translation_libs}
    ${extension_libs}
    ${LLVM_AVAILABLE_LIBS}
    MLIRExecutionEngine
    MLIRBF
)

add_llvm_executable(bfc
  bfc.cpp
  AST.cpp
  MLIRGen.cpp)

llvm_update_compile_flags(bfc)
target_link_libraries(bfc PRIVATE ${LIBS})

mlir_check_all_link_libraries(bfc)
