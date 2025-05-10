cmake_minimum_required(VERSION 3.18)
project(rays LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_ARCHITECTURES 75)  # Change this to match your GPU architecture.

# Set default build type to Debug if not specified.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()

# Find dependencies.
find_package(CUDAToolkit REQUIRED)
find_package(OpenGL      REQUIRED)
find_package(glfw3       REQUIRED)

# Build ImGui as static library.
add_library(imgui STATIC
    external/imgui/imgui.cpp
    external/imgui/imgui_draw.cpp
    external/imgui/imgui_tables.cpp
    external/imgui/imgui_widgets.cpp
    external/imgui/backends/imgui_impl_glfw.cpp
    external/imgui/backends/imgui_impl_opengl3.cpp
)

# Include directories for ImGui.
target_include_directories(imgui PUBLIC
    external/imgui
    external/imgui/backends
)

# Source files.
file(GLOB_RECURSE SOURCES
    "src/main.cpp"
    "src/**/*.cu"
    "src/**/*.cpp"
)

# Create executable.
add_executable(${PROJECT_NAME} ${SOURCES})

# Enable separable compilation for CUDA.
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Include directories for the executable.
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    include
)

# Treat stb as a system include to suppress warnings.
target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE
    external/stb
)

# Link libraries to the executable.
target_link_libraries(${PROJECT_NAME} PRIVATE
    CUDA::cudart
    OpenGL::GL
    glfw
    imgui
)

# C++ compiler flags.
set(CXX_FLAGS_DEBUG
    -g
    -O2
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wconversion
    -Wsign-conversion
)

set(CXX_FLAGS_RELEASE
    -O3
    -Wall
    -Wextra
    -Wpedantic
    -march=native
)

# CUDA compiler flags.
set(CUDA_FLAGS_DEBUG
    -g
    -O2
    --Werror cross-execution-space-call
)

set(CUDA_FLAGS_RELEASE
    -O3
    --use_fast_math
    --Werror cross-execution-space-call
)

# Apply compiler flags based on build type.
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:${CXX_FLAGS_DEBUG}>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:${CXX_FLAGS_RELEASE}>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:${CUDA_FLAGS_DEBUG}>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:${CUDA_FLAGS_RELEASE}>
)

# Print build configuration.
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
