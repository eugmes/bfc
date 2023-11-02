cmake_minimum_required(VERSION 3.20.0)
project(bfc LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})

add_executable(bfc
  bfc.cpp
  parser/AST.cpp
  mlir/MLIRGen.cpp)

target_link_libraries(bfc PRIVATE
  LLVMSupport
  MLIRAnalysis
  MLIRIR
  MLIRParser
  MLIRSideEffectInterfaces
  MLIRTransforms
  MLIRFuncDialect
  MLIRArithDialect
  MLIRIndexDialect
  MLIRSCFDialect
  MLIRMemRefDialect)

target_include_directories(bfc PRIVATE
  include
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS})

target_compile_definitions(bfc PRIVATE ${LLVM_DEFINITIONS_LIST})
