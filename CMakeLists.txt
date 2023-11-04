cmake_minimum_required(VERSION 3.20.0)
project(bfc LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_DIR}")

include(HandleLLVMOptions)
include(AddLLVM)
include(TableGen)
include(AddMLIR)

set(BF_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(BF_BINARY_DIR ${PROJECT_BINARY_DIR})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${BF_SOURCE_DIR}/include)
include_directories(${BF_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)

add_executable(bfc
  bfc.cpp
  parser/AST.cpp
  mlir/MLIRGen.cpp
  mlir/LowerToLLVM.cpp
  mlir/Dialect.cpp)

add_dependencies(bfc MLIRBFOpsIncGen)

include_directories(include)

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
  MLIRFuncAllExtensions
  MLIRMemRefDialect
  MLIRIndexToLLVM
  MLIRArithToLLVM
  MLIRMemRefToLLVM
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRSCFToControlFlow
  MLIRLLVMIRTransforms
  MLIRExecutionEngine
  MLIRTargetLLVMIRImport)
