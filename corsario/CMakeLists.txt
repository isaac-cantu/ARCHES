cmake_minimum_required(VERSION 3.15)
project(corsario CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -------------------------------------------------------------------
# Native executable
# -------------------------------------------------------------------
add_executable(test_corsario include/test.cpp)
target_include_directories(test_corsario PRIVATE include)
target_compile_options(test_corsario PRIVATE -O2 -Wall -Wextra)

# -------------------------------------------------------------------
# pybind11 module (optional — only built if pybind11 is found)
# -------------------------------------------------------------------
find_package(pybind11 QUIET)
if(pybind11_FOUND)
    pybind11_add_module(corsario pybind_corsario.cpp)
    target_include_directories(corsario PRIVATE include)
    target_compile_options(corsario PRIVATE -O2)
    message(STATUS "pybind11 found — Python module 'corsario' will be built")
else()
    message(STATUS "pybind11 not found — skipping Python module")
endif()