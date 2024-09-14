cmake_minimum_required(VERSION 3.18...3.27.8)

project(QR  VERSION 1.0
            DESCRIPTION "Eigenvalue Solver using the QR Algorithm"
            LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(debug_options -Wall -Wextra -Wpedantic -Wfloat-equal -Wshadow -Wunreachable-code -Wdouble-promotion -Wmisleading-indentation)

add_library(QR INTERFACE include/qr.hh)
#target_compile_options(QR INTERFACE -O3 -msse -msse2 -DIMPLICIT -DSINGLE -pg ${debug_options})
target_compile_options(QR INTERFACE -O3 -msse -msse2 -DIMPLICIT -DSINGLE ${debug_options})


find_package(Eigen3 3.3 REQUIRED)
target_link_libraries(QR INTERFACE Eigen3::Eigen)
#find_package(easy_profiler REQUIRED)
#target_link_libraries(QR INTERFACE easy_profiler)

add_subdirectory(evaluation)
add_subdirectory(tests)
