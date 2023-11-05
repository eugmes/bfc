configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(BF_TEST_DEPENDS
        FileCheck count not
        bf-opt
        )

add_lit_testsuite(check-bf "Running the bf regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${BF_TEST_DEPENDS}
        )
set_target_properties(check-bf PROPERTIES FOLDER "Tests")

add_lit_testsuites(BF ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${BF_TEST_DEPENDS})
