# Find Boost
find_package(Boost REQUIRED COMPONENTS iostreams)

# Build examples
file(GLOB EXAMPLE_SOURCES CONFIGURE_DEPENDS *.cpp)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)

foreach(src ${EXAMPLE_SOURCES})
    get_filename_component(name ${src} NAME_WE)
    add_executable(${name} ${src})

    target_include_directories(${name} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/external
    )

    target_link_libraries(${name} PRIVATE
        Boost::iostreams
    )

    set_target_properties(${name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build
    )
endforeach()