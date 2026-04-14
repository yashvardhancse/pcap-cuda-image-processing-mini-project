cmake_minimum_required(VERSION 3.24)
project(cuda_image_processing LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75;80;86;89 CACHE STRING "CUDA architectures" FORCE)

find_package(CUDAToolkit REQUIRED)

add_library(cuda_image_processing SHARED
    gaussian_blur.cu
    sharpening.cu
    sobel_edge.cu
    dilation.cu
    erosion.cu
)

target_include_directories(cuda_image_processing
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(cuda_image_processing PRIVATE CUDA::cudart)
set_target_properties(cuda_image_processing PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
