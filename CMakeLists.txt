cmake_minimum_required(VERSION 3.18)
project(rays LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_ARCHITECTURES 75)  # Change this to match your GPU architecture

find_package(PNG         REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenGL      REQUIRED)
find_package(glfw3       REQUIRED)

add_library(imgui STATIC
    external/imgui/imgui.cpp
    external/imgui/imgui_draw.cpp
    external/imgui/imgui_tables.cpp
    external/imgui/imgui_widgets.cpp
    external/imgui/backends/imgui_impl_glfw.cpp
    external/imgui/backends/imgui_impl_opengl3.cpp
)

target_include_directories(imgui PUBLIC
    external/imgui
    external/imgui/backends
)

include_directories(
    ${CUDAToolkit_INCLUDE_DIRS}
    ${Imgui_INCLUDE_DIRS}
    include
    external/stb
)

file(GLOB_RECURSE SOURCES "src/*.cu")

add_executable(${PROJECT_NAME} ${SOURCES})

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(${PROJECT_NAME} PRIVATE
    CUDA::cudart
    OpenGL::GL
    glfw
    imgui
)

target_compile_options(${PROJECT_NAME} 
    PRIVATE
    "-g"
    "-O2"
)
