cmake_minimum_required(VERSION 3.20)
project(lob_market_maker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
add_compile_options(-Wall -Wextra -Wpedantic)

# Core engine (no Python dependency).
add_library(lobcore STATIC
  engine/cpp/order_book.cpp
  engine/cpp/orders.cpp
  engine/cpp/matching.cpp)
target_include_directories(lobcore PUBLIC engine/cpp)
set_target_properties(lobcore PROPERTIES POSITION_INDEPENDENT_CODE ON)

# C++ unit + fuzz tests: `ctest --test-dir build`
enable_testing()
add_executable(lob_tests engine/cpp/tests/test_engine.cpp)
target_link_libraries(lob_tests PRIVATE lobcore)
add_test(NAME lob_engine_tests COMMAND lob_tests)

# Python extension `engine._lob_cpp` (pybind11). Needs: -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
option(LOB_BUILD_PYTHON "Build the pybind11 extension" ON)
if(LOB_BUILD_PYTHON)
  set(PYBIND11_FINDPYTHON ON)  # honour -DPython_EXECUTABLE so the extension matches the venv's ABI
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_lob_cpp engine/cpp/bindings.cpp)
  target_link_libraries(_lob_cpp PRIVATE lobcore)
  set_target_properties(_lob_cpp PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/engine)
endif()
