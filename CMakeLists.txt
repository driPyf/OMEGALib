# Copyright 2025-present the zvec project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

# Find OpenMP (required by LightGBM)
find_package(OpenMP REQUIRED)

# LightGBM build options
set(BUILD_STATIC_LIB ON CACHE BOOL "Build LightGBM as static library" FORCE)
set(BUILD_CLI OFF CACHE BOOL "Disable LightGBM CLI" FORCE)
set(USE_OPENMP ON CACHE BOOL "Enable OpenMP for LightGBM" FORCE)

# Add LightGBM as subdirectory
add_subdirectory(lightgbm EXCLUDE_FROM_ALL)

# ============================================================
# Omega source files
# ============================================================
set(OMEGA_SOURCES
  src/feature_extractor.cpp
  src/model_manager.cpp
  src/search_context.cpp
  src/omega_api.cpp
  src/omega_trainer.cpp
  src/tree_inference.cpp
)

# Create omega library
add_library(omega STATIC ${OMEGA_SOURCES})
target_include_directories(omega PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lightgbm/include>
  $<INSTALL_INTERFACE:include>
)

# Link LightGBM and OpenMP
target_link_libraries(omega PUBLIC _lightgbm OpenMP::OpenMP_CXX)

# Restore output directory
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${_SAVED_CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
unset(_SAVED_CMAKE_ARCHIVE_OUTPUT_DIRECTORY)

# Set include directory for parent scope
get_target_property(omega_SOURCE_DIR omega SOURCE_DIR)
set(OMEGA_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(omega PUBLIC $<BUILD_INTERFACE:${OMEGA_INCLUDE_DIR}>)
