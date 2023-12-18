add_executable(evaluate main.cc test.cc)

target_include_directories(evaluate PUBLIC ../include)
target_link_libraries(evaluate Eigen3::Eigen)
target_link_libraries(evaluate QR)
