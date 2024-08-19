cmake_minimum_required(VERSION 3.18...3.27.8)

project(QR  VERSION 1.0
            DESCRIPTION "Eigenvalue Solver using the QR Algorithm"
            LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


add_library(QR INTERFACE include/qr.hh)
target_compile_options(QR INTERFACE -Wall -Wextra -Wpedantic -Wfloat-equal -Wshadow -Wunreachable-code -Wdouble-promotion -Wmisleading-indentation)
#target_compile_options(QR INTERFACE -Wall -Wextra)

#find_package(easy_profiler REQUIRED)
#
#target_link_libraries(QR easy_profiler)

find_package(Eigen3 3.3)

add_subdirectory(evaluation)

add_subdirectory(tests)


#find_package(Catch2 3 REQUIRED)

#add_executable(tests test.cpp)
#target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)
#
#add_executable(custom-main-tests test.cpp test-main.cpp)
#target_link_libraries(custom-main-tests PRIVATE Catch2::Catch2)

#target_compile_features(QR PUBLIC cxx_std_17)

#add_executable(evaluate main.cc)
#target_link_libraries(evaluate PUBLIC QR) # very unsure if this is correct

#target_include_directories(evaluate PUBLIC ../helpfunctions)

# Current state in the tutorial




#DEBUG := -g -Wall -Wextra -Wfloat-equal -Wshadow -Wunreachable-code -fsanitize=address -fsanitize=undefined -fsanitize=pointer-compare -Wunused-but-set-variable -pedantic


#target_compile_options(${PROJECT_NAME} PUBLIC -llapack -DVERSION=1)

