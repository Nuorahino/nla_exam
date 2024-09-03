add_executable(evaluate main.cc test.cc)

set(debug_options "-Wall -Wextra -Wpedantic -Wfloat-equal -Wshadow -Wunreachable-code -Wdouble-promotion -Wmisleading-indentation")

target_include_directories(evaluate PUBLIC ../include)
find_package(Eigen3 3.3 REQUIRED)
target_link_libraries(evaluate Eigen3::Eigen)
#find_package(easy_profiler REQUIRED)
#target_link_libraries(evaluate easy_profiler)
find_package(lapack REQUIRED)
target_link_libraries(evaluate lapack)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MY_FLAGS} ${debug_options} -O3 -msse -msse2")

#target_link_libraries(evaluate QR)
#target_compile_options(evaluate PRIVATE -O3 -march=native)
