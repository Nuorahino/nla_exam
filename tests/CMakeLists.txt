find_package(Catch2 3 REQUIRED)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/tests/testfiles)

add_executable(test_helpfunctions test_helpfunctions.cpp)
target_link_libraries(test_helpfunctions PRIVATE Catch2::Catch2WithMain)
target_compile_options(test_helpfunctions PRIVATE -DBINARY_DIR="${CMAKE_BINARY_DIR}/tests/testfiles")
add_executable(test_givens test_givens.cpp)
target_link_libraries(test_givens PRIVATE Catch2::Catch2WithMain)
target_link_libraries(test_givens PRIVATE QR)
