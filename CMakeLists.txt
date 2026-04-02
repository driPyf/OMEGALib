# Copyright 2025-present Yifan Peng
#
# Licensed under the MIT License. See the LICENSE file in the project root for
# license terms.

# Disable tests for omega when building as part of zvec
set(OMEGA_BUILD_TESTS OFF CACHE BOOL "Disable testing in omega" FORCE)

# Set output directory to match other thirdparty libraries
set(_SAVED_CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${EXTERNAL_LIB_DIR})

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ============================================================
# LightGBM dependency (always required for omega)
# ============================================================
message(STATUS "OMEGA: Building with LightGBM support")

# Find OpenMP when available. Some platforms, such as GitHub-hosted macOS
# runners, do not provide it out of the box. In that case we still build
# LightGBM/OMEGA in single-threaded mode so zvec remains portable.
find_package(OpenMP QUIET)
if(OpenMP_FOUND)
  set(OMEGA_USE_OPENMP ON)
  message(STATUS "OMEGA: OpenMP found, enabling multi-threaded LightGBM")
else()
  set(OMEGA_USE_OPENMP OFF)
  message(WARNING "OMEGA: OpenMP not found, building LightGBM without OpenMP")
endif()

# LightGBM build options
set(BUILD_STATIC_LIB ON CACHE BOOL "Build LightGBM as static library" FORCE)
set(BUILD_CLI OFF CACHE BOOL "Disable LightGBM CLI" FORCE)
set(USE_OPENMP ${OMEGA_USE_OPENMP} CACHE BOOL "Enable OpenMP for LightGBM" FORCE)

# Add LightGBM as subdirectory
add_subdirectory(lightgbm EXCLUDE_FROM_ALL)

# Omega only uses single-node LightGBM training. Disable the vendored socket
# networking backend at the target level so platform builds (notably Android)
# do not pull in distributed networking code paths.
set(_OMEGA_LIGHTGBM_DISABLE_SOCKET_FLAGS
  $<$<CXX_COMPILER_ID:MSVC>:/UUSE_SOCKET>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-UUSE_SOCKET>)
set(_OMEGA_LIGHTGBM_STUB_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/src/lightgbm_network_stubs.cpp)
target_compile_options(lightgbm_objs PRIVATE ${_OMEGA_LIGHTGBM_DISABLE_SOCKET_FLAGS})

get_target_property(_OMEGA_LIGHTGBM_SOURCES lightgbm_objs SOURCES)
list(FILTER _OMEGA_LIGHTGBM_SOURCES EXCLUDE REGEX "(^|.*/)src/network/linkers_socket\\.cpp$")
set_property(TARGET lightgbm_objs PROPERTY SOURCES "${_OMEGA_LIGHTGBM_SOURCES}")
target_sources(_lightgbm PRIVATE ${_OMEGA_LIGHTGBM_STUB_SRC})
set_source_files_properties(
  ${_OMEGA_LIGHTGBM_STUB_SRC}
  PROPERTIES COMPILE_OPTIONS "${_OMEGA_LIGHTGBM_DISABLE_SOCKET_FLAGS}")
unset(_OMEGA_LIGHTGBM_SOURCES)
unset(_OMEGA_LIGHTGBM_DISABLE_SOCKET_FLAGS)
unset(_OMEGA_LIGHTGBM_STUB_SRC)

# ============================================================
# Eigen dependency (header-only, for fast ground truth computation)
# ============================================================
message(STATUS "OMEGA: Using Eigen for fast ground truth computation")

# Eigen is header-only, just add include directory
set(EIGEN_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/eigen)
if(NOT EXISTS ${EIGEN_INCLUDE_DIR}/Eigen/Core)
  message(FATAL_ERROR "Eigen not found at ${EIGEN_INCLUDE_DIR}. Please run: git submodule update --init --recursive")
endif()

# ============================================================
# Omega source files
# ============================================================
set(OMEGA_SOURCES
  src/model_manager.cpp
  src/search_context.cpp
  src/omega_api.cpp
  src/omega_trainer.cpp
  src/tree_inference.cpp
  src/ground_truth.cpp
)

# Create omega library
add_library(omega STATIC ${OMEGA_SOURCES})
target_include_directories(omega PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lightgbm/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lightgbm/external_libs/fast_double_parser/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lightgbm/external_libs/fmt/include>
  $<BUILD_INTERFACE:${EIGEN_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>
)
target_compile_options(omega PUBLIC
  $<$<CXX_COMPILER_ID:MSVC>:/utf-8>
)

# Link LightGBM and OpenMP when available
target_link_libraries(omega PUBLIC _lightgbm)
if(OMEGA_USE_OPENMP)
  target_link_libraries(omega PUBLIC OpenMP::OpenMP_CXX)
endif()

# Restore output directory
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${_SAVED_CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
unset(_SAVED_CMAKE_ARCHIVE_OUTPUT_DIRECTORY)

# Set include directory for parent scope
get_target_property(omega_SOURCE_DIR omega SOURCE_DIR)
set(OMEGA_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(omega PUBLIC $<BUILD_INTERFACE:${OMEGA_INCLUDE_DIR}>)

if(BUILD_TESTING)
  enable_testing()

  add_executable(omega_model_manager_test tests/model_manager_test.cpp)
  target_link_libraries(omega_model_manager_test PRIVATE omega gtest gtest_main)
  target_include_directories(omega_model_manager_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  add_test(NAME omega_model_manager_test COMMAND omega_model_manager_test)
endif()
