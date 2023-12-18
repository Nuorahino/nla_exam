find_package(Catch2 3 REQUIRED)

add_executable(simple_test test_Hello.cpp)
target_link_libraries(simple_test PRIVATE Catch2::Catch2WithMain)
