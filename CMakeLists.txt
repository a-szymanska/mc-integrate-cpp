cmake_minimum_required(VERSION 3.10)
project(mc_integrate_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_GNUPLOT "Enable gnuplot" OFF)
option(USE_OMP "Enable OpenMP" OFF)

if(USE_OMP AND APPLE)
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY /opt/homebrew/opt/libomp/lib/libomp.dylib)
    include_directories(/opt/homebrew/opt/libomp/include)
endif()

find_package(OpenMP)


# External dependencies
if(USE_GNUPLOT)
    set(GNUPLOT_HEADER_DIR ${CMAKE_BINARY_DIR}/external)
    file(MAKE_DIRECTORY ${GNUPLOT_HEADER_DIR})

    file(DOWNLOAD
        https://raw.githubusercontent.com/dstahlke/gnuplot-iostream/master/gnuplot-iostream.h
        ${GNUPLOT_HEADER_DIR}/gnuplot-iostream.h
    )
endif()


# Source code
add_library(integrate
    src/integrate_quadrature.cpp
)

target_include_directories(integrate PUBLIC include)

add_library(sample INTERFACE)
target_include_directories(sample INTERFACE include)

target_link_libraries(integrate PUBLIC sample)

# OpenMP
if(USE_OMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(integrate PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(integrate PUBLIC USE_OMP)
endif()


# Linking
target_link_libraries(integrate PUBLIC sample)

if(USE_OMP AND OpenMP_CXX_FOUND)
    target_link_libraries(integrate PUBLIC OpenMP::OpenMP_CXX)
endif()


# Tests
enable_testing()

add_executable(test_integrate_mc tests/test_integrate_mc.cpp)
target_link_libraries(test_integrate_mc integrate)

add_executable(test_integrate_quadrature tests/test_integrate_quadrature.cpp)
target_link_libraries(test_integrate_quadrature integrate)

add_executable(test_sample tests/test_sample.cpp)
target_link_libraries(test_sample sample)

add_test(NAME test_integrate_mc COMMAND test_integrate_mc)
add_test(NAME test_integrate_quadrature COMMAND test_integrate_quadrature)
add_test(NAME test_sample COMMAND test_sample)


# Subdirectories
add_subdirectory(examples)