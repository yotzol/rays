cmake_minimum_required(VERSION 3.18)
project(rays LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_ARCHITECTURES 75)  # Change this to match your GPU architecture

find_package(PNG REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories(
    ${PNG_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    include
)

file(GLOB_RECURSE SOURCES "src/*.cu")
add_executable(${PROJECT_NAME} ${SOURCES})

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(${PROJECT_NAME}
    ${PNG_LIBRARIES}
    CUDA::cudart
)

target_compile_options(${PROJECT_NAME} 
    PRIVATE
    "-g"
    "-O2"
)
