cmake_minimum_required(VERSION 3.10)
project(mc_integrate_cpp LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(APPLE)
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY /opt/homebrew/opt/libomp/lib/libomp.dylib)
    include_directories(/opt/homebrew/opt/libomp/include)
endif()

find_package(OpenMP)

# Source code
add_library(integrate src/integrate_mc.cpp src/integrate_mc_omp.cpp src/integrate_quadrature.cpp)
target_include_directories(integrate PUBLIC include)
add_library(sample INTERFACE)
target_include_directories(sample INTERFACE include)

target_link_libraries(integrate PUBLIC OpenMP::OpenMP_CXX sample)

# Tests
enable_testing()

add_executable(test_integrate_mc tests/test_integrate_mc.cpp)
target_link_libraries(test_integrate_mc integrate)

add_executable(test_integrate_quadrature tests/test_integrate_quadrature.cpp)
target_link_libraries(test_integrate_quadrature integrate)

add_executable(test_sample tests/test_sample.cpp)
target_link_libraries(test_sample sample)

# if(OpenMP_CXX_FOUND)
#     add_executable(test_integrate_mc_omp tests/test_integrate_mc_omp.cpp)
#     target_link_libraries(test_integrate_mc_omp integrate)
#     add_test(NAME test_integrate_mc_omp COMMAND test_integrate_mc_omp)
# endif()

add_test(NAME test_integrate_mc COMMAND test_integrate_mc)
add_test(NAME test_integrate_quadrature COMMAND test_integrate_quadrature)
add_test(NAME test_sample COMMAND test_sample)
