# select backend
set(FZGPU_BACKEND "CUDA" CACHE STRING "Select the desired backend: CUDA/cuda, HIP/hip and ONEAPI/1api.")

if(FZGPU_BACKEND STREQUAL "CUDA" OR FZGPU_BACKEND STREQUAL "cuda")
  cmake_minimum_required(VERSION 3.18)
  project(CUSZ LANGUAGES CUDA CXX C VERSION 0.1.0)
elseif(FZGPU_BACKEND STREQUAL "HIP" OR FZGPU_BACKEND STREQUAL "hip")
  cmake_minimum_required(VERSION 3.21)
  project(HIPSZ LANGUAGES HIP CXX C VERSION 0.1.0)
elseif(FZGPU_BACKEND STREQUAL "ONEAPI" OR FZGPU_BACKEND STREQUAL "oneapi" OR FZGPU_BACKEND STREQUAL "1api")
  cmake_minimum_required(VERSION 3.23)
  project(DPSZ LANGUAGES CXX C VERSION 0.1.0)
endif()

# general cmake setup
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(BUILD_SHARED_LIBS "prefer shared libaries" ON)

# example and testing
option(FZGPU_BUILD_EXAMPLES "build example codes" ON)


if(FZGPU_BACKEND STREQUAL "CUDA" OR FZGPU_BACKEND STREQUAL "cuda")
  message("[psz::info] CUDA backend has been selected.")
  include(cmake/cuda.cmake)
elseif(FZGPU_BACKEND STREQUAL "HIP" OR FZGPU_BACKEND STREQUAL "hip")
  message("[psz::info] HIP-ROCm backend has been selected.")
  include(cmake/hip.cmake)
elseif(FZGPU_BACKEND STREQUAL "ONEAPI" OR FZGPU_BACKEND STREQUAL "oneapi" OR FZGPU_BACKEND STREQUAL "1api")
  message("[psz::info] oneAPI backend has been selected.")
  include(cmake/dpcpp.cmake)
else()
  message(FATAL_ERROR "Unsupported FZGPU_BACKEND selected: ${FZGPU_BACKEND}")
endif()