add_executable(evaluate main.cc test.cc)

find_package(easy_profiler REQUIRED)
target_include_directories(evaluate PUBLIC ../include)
target_link_libraries(evaluate Eigen3::Eigen)
target_link_libraries(evaluate QR)
target_link_libraries(evaluate easy_profiler)
target_compile_options(evaluate PRIVATE -O3 -g -msse -msse2)
#target_compile_options(evaluate PRIVATE -O3 -march=native)
