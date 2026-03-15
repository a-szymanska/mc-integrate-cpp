cmake_minimum_required(VERSION 3.10)

project(mc_integrate_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

add_library(integrate src/integrate_mc.cpp src/integrate_quadrature.cpp)
target_include_directories(integrate PUBLIC include)

# Tests
add_executable(test_integrate_mc tests/test_integrate_mc.cpp)
target_link_libraries(test_integrate_mc integrate)

add_executable(test_integrate_quadrature tests/test_integrate_quadrature.cpp)
target_link_libraries(test_integrate_quadrature integrate)

add_test(NAME test_integrate_mc COMMAND test_integrate_mc)
add_test(NAME test_integrate_quadrature COMMAND test_integrate_quadrature)