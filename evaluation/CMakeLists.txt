add_executable(evaluate main.cc test.cc)
set(debug_options "-Wall -Wextra -Wpedantic -Wfloat-equal -Wshadow -Wunreachable-code -Wdouble-promotion -Wmisleading-indentation ")

find_package(easy_profiler REQUIRED)
target_include_directories(evaluate PUBLIC ../include)
target_link_libraries(evaluate Eigen3::Eigen)
target_link_libraries(evaluate QR)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MY_FLAGS} ${debug_options} -O3 -msse -msse2 -pg")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MY_FLAGS} ${debug_options} -O3 -msse -msse2")

#target_compile_options(evaluate PRIVATE -O3 -march=native)
