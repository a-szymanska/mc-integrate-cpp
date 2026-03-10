cmake_minimum_required(VERSION 3.10)

project(mc_integrate_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Test executable
add_executable(test_integrate
    tests/test_integrate.cpp
)
target_include_directories(test_integrate
    PRIVATE
    include
)

enable_testing()

add_test(
    NAME run_tests
    COMMAND test_integrate
)