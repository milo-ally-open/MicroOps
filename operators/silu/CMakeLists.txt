cmake_minimum_required(VERSION 3.21)
project(silu VERSION 0.1.0 LANGUAGES CXX)

option(OPS_ENABLE_CUDA "Build the silu CUDA backend" OFF)
option(OPS_ENABLE_CANN "Build the silu CANN backend" OFF)
option(OPS_BUILD_EXAMPLES "Build silu examples" OFF)

if(NOT OPS_ENABLE_CUDA AND NOT OPS_ENABLE_CANN)
    message(FATAL_ERROR "No device backend selected. Enable OPS_ENABLE_CUDA or OPS_ENABLE_CANN.")
endif()

if(OPS_ENABLE_CUDA AND OPS_ENABLE_CANN)
    message(FATAL_ERROR "Configure one backend at a time so each build is isolated and reproducible.")
endif()

if(OPS_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR
            "OPS_ENABLE_CUDA=ON, but no CUDA compiler was found. Install a compatible CUDA Toolkit/compiler.")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_subdirectory(src/cuda)

    if(OPS_BUILD_EXAMPLES)
        add_executable(silu_demo examples/cuda/silu_demo.cu)
        target_include_directories(silu_demo PRIVATE src/cuda/include)
        target_link_libraries(silu_demo PRIVATE silu_cuda CUDA::cudart)
        set_target_properties(silu_demo PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            CUDA_EXTENSIONS OFF)
    endif()
endif()

if(OPS_ENABLE_CANN)
    add_subdirectory(src/cann)
endif()
