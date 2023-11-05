get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRBF
        )
add_llvm_executable(bf-opt bf-opt.cpp)

llvm_update_compile_flags(bf-opt)
target_link_libraries(bf-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(bf-opt)
