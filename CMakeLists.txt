cmake_minimum_required(VERSION 3.27)
project(RadFoam VERSION 1.0.0)

cmake_policy(SET CMP0060 NEW)

enable_language(CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(GPU_DEBUG
    ON
    CACHE BOOL "Enable GPU debug features")
add_definitions(-DGPU_DEBUG=$<BOOL:${GPU_DEBUG}>)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      "Release"
      CACHE STRING "Build type")
endif()

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/external"
     "${CMAKE_SOURCE_DIR}/external/submodules")

find_package(
  Python3
  COMPONENTS Interpreter Development.Module
  REQUIRED)

find_package(pybind11 REQUIRED)

if(NOT Torch_DIR)
  set(Torch_DIR ${Python3_SITELIB}/torch/share/cmake/Torch)
endif()

find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH
             "${TORCH_INSTALL_PREFIX}/lib")

add_subdirectory(external)
include_directories(${RADFOAM_EXTERNAL_INCLUDES})

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX
      "${CMAKE_SOURCE_DIR}/radfoam"
      CACHE PATH "..." FORCE)
endif()

set(RADFOAM_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})

add_subdirectory(src)
add_subdirectory(torch_bindings)
