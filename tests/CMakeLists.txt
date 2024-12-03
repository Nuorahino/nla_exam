find_package(Catch2 3 REQUIRED)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/tests/testfiles)

add_executable(test_helpfunctions test_helpfunctions.cpp)
target_link_libraries(test_helpfunctions PRIVATE Catch2::Catch2WithMain)
target_compile_options(test_helpfunctions PRIVATE -DBINARY_DIR="${CMAKE_BINARY_DIR}/tests/testfiles")

find_package(easy_profiler REQUIRED)
find_package(lapack REQUIRED)

add_executable(test_givens test_givens.cpp)
target_link_libraries(test_givens PRIVATE Catch2::Catch2WithMain)
#target_link_libraries(test_givens PRIVATE QR)
target_link_libraries(test_givens PRIVATE easy_profiler)
target_link_libraries(test_givens PRIVATE lapack)

add_executable(test_householder test_householder.cpp)
target_link_libraries(test_householder PRIVATE Catch2::Catch2WithMain)
#target_link_libraries(test_householder PRIVATE QR)
target_link_libraries(test_householder PRIVATE easy_profiler)
target_link_libraries(test_householder PRIVATE lapack)

add_executable(test_similiarity_transform test_similiarity_transform.cpp)
target_link_libraries(test_similiarity_transform PRIVATE Catch2::Catch2WithMain)
#target_link_libraries(test_similiarity_transform PRIVATE QR)
target_link_libraries(test_similiarity_transform PRIVATE easy_profiler)
target_link_libraries(test_similiarity_transform PRIVATE lapack)

add_executable(test_benchmarks test_benchmarks.cpp)
target_link_libraries(test_benchmarks PRIVATE Catch2::Catch2WithMain)
#target_link_libraries(test_benchmarks PRIVATE QR)
target_link_libraries(test_benchmarks PRIVATE easy_profiler)
target_link_libraries(test_benchmarks PRIVATE lapack)

add_executable(test_tests test_tests.cpp)
target_link_libraries(test_tests PRIVATE Catch2::Catch2WithMain)
