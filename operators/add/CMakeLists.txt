cmake_minimum_required(VERSION 3.21)
project(add VERSION 0.1.0 LANGUAGES CXX)

option(OPS_ENABLE_CUDA "Build the add CUDA backend" OFF)
option(OPS_ENABLE_CANN "Build the add CANN backend" OFF)
option(OPS_BUILD_EXAMPLES "Build add examples" OFF)

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
    add_subdirectory(cuda)

    if(OPS_BUILD_EXAMPLES)
        add_executable(add_demo examples/add_demo.cu)
        target_include_directories(add_demo PRIVATE cuda/include)
        target_link_libraries(add_demo PRIVATE add_cuda CUDA::cudart)
        set_target_properties(add_demo PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            CUDA_EXTENSIONS OFF)
    endif()
endif()

if(OPS_ENABLE_CANN)
    add_subdirectory(cann)
endif()
