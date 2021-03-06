# CMake version
cmake_minimum_required(VERSION 3.8.2 FATAL_ERROR)

# Include cmake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Builds/CMake")

# Declare project
project(VulkanEngine)

# Set output directories
set(DEFAULT_CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

# Find vulkan sdk.
message(STATUS "Attempting auto locate Vulkan using CMake......")

# Find Vulkan Path using CMake's Vulkan Module
# This will return Boolean 'Vulkan_FOUND' indicating the status of find as success(ON) or fail(OFF).
# Include directory path - 'Vulkan_INCLUDE_DIRS' and 'Vulkan_LIBRARY' with required libraries.
find_package(Vulkan REQUIRED)

# Try extracting VulkanSDK path from ${Vulkan_INCLUDE_DIRS}
if (NOT ${Vulkan_INCLUDE_DIRS} STREQUAL "")
	set(VULKAN_PATH ${Vulkan_INCLUDE_DIRS})
	STRING(REGEX REPLACE "/Include" "" VULKAN_PATH ${VULKAN_PATH})
endif()
	 
if(NOT Vulkan_FOUND)
	# CMake may fail to locate the libraries but could be able to 
	# provide some path in Vulkan SDK include directory variable
	# 'Vulkan_INCLUDE_DIRS', try to extract path from this.
	message(STATUS "Failed to locate Vulkan SDK, retrying again...")
	if(EXISTS "${VULKAN_PATH}")
		message(STATUS "Successfully located the Vulkan SDK: ${VULKAN_PATH}")
	else()
		message("Error: Unable to locate Vulkan SDK. Please turn off auto locate option by specifying 'AUTO_LOCATE_VULKAN' as 'OFF'")
		message("and specify manually path using 'VULKAN_SDK' and 'VULKAN_VERSION' variables in the CMakeLists.txt.")
		return()
	endif()
endif()

# vulkan-1 library for build Vulkan application.
# set(VULKAN_LIB_LIST "vulkan-1")

if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
	# Include Vulkan header files from Vulkan SDK
	include_directories(AFTER ${VULKAN_PATH}/Include)

	# Link directory for vulkan-1
	link_directories(${VULKAN_PATH}/Bin;${VULKAN_PATH}/Lib;)
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
else()
endif()

include_directories(${SDL_PATH}/include)

if("${CMAKE_SIZEOF_VOID_P}" STREQUAL "4")
	set(SDL2_LINK_DIRECTORY ${SDL_PATH}/lib/x86)
else()
	set(SDL2_LINK_DIRECTORY ${SDL_PATH}/lib/x64)
endif()
link_directories(${SDL2_LINK_DIRECTORY})

find_file(SDL2_DLL "SDL2.dll" HINTS ${SDL2_LINK_DIRECTORY})

configure_file(${SDL2_DLL} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/SDL2.dll COPYONLY)

# Includes
include_directories(Includes)
include_directories(Libraries)
include_directories(Libraries/glm)
include_directories(Libraries/imgui)
include_directories(Libraries/stb_image)
include_directories(Libraries/tinyobjloader)
include_directories(Libraries/vkbootstrap)
include_directories(Libraries/vma)
include_directories(Libraries/volk)

# Set enable output of compile commands during generation
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type - Release by default
message("CMake build type: " ${CMAKE_BUILD_TYPE})
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
elseif(CMAKE_BUILD_TYPE MATCHES "Debug")
    add_definitions(-DDEBUG)
endif()

# Compile options
include(Builds/CMake/CompileOptions.cmake)

# Overrides
set(CMAKE_MACOSX_RPATH ON)

# Set resources
set(RESOURCES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Resources)

add_subdirectory(Libraries/imgui)
add_subdirectory(Libraries/tinyobjloader)
add_subdirectory(Libraries/vkbootstrap)
add_subdirectory(Libraries/volk)

# find glslangValidator program
find_program(GLSL_VALIDATOR glslangValidator HINTS /usr/bin /usr/local/bin $ENV{VULKAN_SDK}/Bin/ $ENV{VULKAN_SDK}/Bin32/)

# find all the shader files under the shaders folder.
file(GLOB_RECURSE GLSL_SOURCE_FILES
	"${RESOURCES_DIR}/shaders/*.frag",
	"${RESOURCES_DIR}/shaders/*.vert",
	"${RESOURCES_DIR}/shaders/*.comp"
)

# iterate each shader and compile them
foreach(GLSL ${GLSL_SOURCE_FILES})
	message(STATUS "Building Shader")
	get_filename_component(FILE_NAME ${GLSL} NAME)
	set(SPIRV "${RESOURCES_DIR}/shaders/${FILE_NAME}.spv")
	message(STATUS ${GLSL})
	add_custom_command(
		OUTPUT ${SPIRV}
		COMMAND ${GLSL_VALIDATOR} -V ${GLSL} -o ${SPIRV}
		DEPENDS ${GLSL}
	)
	list(APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach(GLSL)

add_custom_target(
	run ALL
	DEPENDS ${SPIRV_BINARY_FILES}
)

add_subdirectory(Sources)