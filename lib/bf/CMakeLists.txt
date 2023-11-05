add_mlir_dialect_library(MLIRBF
    BFTypes.cpp
    BFDialect.cpp
    BFOps.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/bf

    DEPENDS
    MLIRBFOpsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRInferTypeOpInterface
    )
