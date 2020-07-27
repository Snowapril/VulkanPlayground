# CMake version
cmake_minimum_required(VERSION 3.8.2 FATAL_ERROR)

# Include cmake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Builds/CMake")

# Declare project
project(VulkanPlayground)

# Set output directories
set(DEFAULT_CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

# Includes
include_directories(Libraries)
include_directories(Libraries/glm)
include_directories(Libraries/glfw/include)
include_directories(${VULKAN_SDK_PATH}/include)
link_directories(${VULKAN_SDK_PATH}/lib)

# Set enable output of compile commands during generation
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type - Release by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compile options
include(Builds/CMake/CompileOptions.cmake)

# Overrides
set(CMAKE_MACOSX_RPATH ON)

option(GLFW_USE_OSMESA "GLFW OSMesa for headless environment" OFF)
if (GLFW_USE_OSMESA)
    add_definitions(-DCUBBYFLOW_HEADLESS)
endif()
add_subdirectory(Libraries/glfw)
add_subdirectory(Libraries/glm)
add_subdirectory(vulkan-tutorial)