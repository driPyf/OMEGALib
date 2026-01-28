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

cmake_minimum_required(VERSION 3.15)
project(omega VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags for Google C++ style
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Option to enable/disable OMEGA
option(ZVEC_ENABLE_OMEGA "Enable OMEGA adaptive search" ON)

if(ZVEC_ENABLE_OMEGA)
  # Include directories
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

  # Source files
  set(OMEGA_SOURCES
    src/tree_inference.cpp
    src/feature_extractor.cpp
    src/model_manager.cpp
    src/search_context.cpp
    src/omega_api.cpp
  )

  # Create omega library
  add_library(omega STATIC ${OMEGA_SOURCES})
  target_include_directories(omega PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )

  # Enable testing
  option(OMEGA_BUILD_TESTS "Build OMEGA tests" ON)

  if(OMEGA_BUILD_TESTS)
    enable_testing()

    # Find GoogleTest
    find_package(GTest REQUIRED)
    include_directories(${GTEST_INCLUDE_DIRS})

    # Test executables
    add_executable(tree_inference_test
      tests/tree_inference_test.cpp
    )
    target_link_libraries(tree_inference_test
      omega
      GTest::GTest
      GTest::Main
    )
    add_test(NAME tree_inference_test COMMAND tree_inference_test)

    add_executable(feature_extractor_test
      tests/feature_extractor_test.cpp
    )
    target_link_libraries(feature_extractor_test
      omega
      GTest::GTest
      GTest::Main
    )
    add_test(NAME feature_extractor_test COMMAND feature_extractor_test)

    add_executable(search_context_test
      tests/search_context_test.cpp
    )
    target_link_libraries(search_context_test
      omega
      GTest::GTest
      GTest::Main
    )
    add_test(NAME search_context_test COMMAND search_context_test)
  endif()

  # Installation
  install(TARGETS omega
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
  )

  install(DIRECTORY include/
    DESTINATION include/omega
    FILES_MATCHING PATTERN "*.h"
  )
endif()
