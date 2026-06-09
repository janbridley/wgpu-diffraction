cmake_minimum_required(VERSION 3.15)
project(sf3d_gpu)

find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)

# Try to find nanobind
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${nanobind_DIR}")
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(sf3d_gpu src/main.cpp)

install(TARGETS sf3d_gpu LIBRARY DESTINATION .)

if(APPLE)
    find_library(OPENCL_LIBRARIES OpenCL REQUIRED)
else()
    find_package(OpenCL REQUIRED)
endif()

target_include_directories(sf3d_gpu PRIVATE src)
target_link_libraries(sf3d_gpu PRIVATE ${OPENCL_LIBRARIES})
