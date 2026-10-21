cmake_minimum_required(VERSION 3.21)
project(gelu_cuda_tests VERSION 0.1.0 LANGUAGES CXX)

include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR
        "The gelu CUDA test project requires nvcc/CUDA Toolkit. Install CUDA or run a backend available on this host.")
endif()
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
include(CTest)

get_filename_component(GELU_OPERATOR_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
add_subdirectory("${GELU_OPERATOR_ROOT}/src/cuda" "${CMAKE_CURRENT_BINARY_DIR}/operator_cuda")

set(OPS_TEST_CORRECTNESS_SOURCE "${CMAKE_CURRENT_LIST_DIR}/correctness/gelu_cuda_test.cu"
    CACHE FILEPATH "CUDA correctness test source")
set(OPS_TEST_MEMORY_SOURCE "${CMAKE_CURRENT_LIST_DIR}/memory/gelu_cuda_memory_eval.cu"
    CACHE FILEPATH "CUDA memory evaluation source")
set(OPS_TEST_PERFORMANCE_SOURCE "${CMAKE_CURRENT_LIST_DIR}/performance/gelu_cuda_benchmark.cu"
    CACHE FILEPATH "CUDA performance evaluation source")

add_executable(gelu_cuda_correctness_test "${OPS_TEST_CORRECTNESS_SOURCE}")
target_link_libraries(gelu_cuda_correctness_test PRIVATE gelu_cuda CUDA::cudart)
target_compile_definitions(gelu_cuda_correctness_test PRIVATE
    GELU_CUDA_KERNEL_FILE="${OPS_KERNEL_FILE}"
    GELU_CUDA_HOST_FILE="${OPS_HOST_FILE}")
set_target_properties(gelu_cuda_correctness_test PROPERTIES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF)
add_test(NAME gelu_cuda_correctness COMMAND gelu_cuda_correctness_test)
set_tests_properties(gelu_cuda_correctness PROPERTIES SKIP_RETURN_CODE 77)

set(GELU_CUDA_MEMORY_RESULTS "${GELU_OPERATOR_ROOT}/tests/cuda/memory/results/memory.csv"
    CACHE FILEPATH "CSV path for real CUDA memory-evaluation records")
add_executable(gelu_cuda_memory_eval "${OPS_TEST_MEMORY_SOURCE}")
target_link_libraries(gelu_cuda_memory_eval PRIVATE CUDA::cudart)
target_compile_definitions(gelu_cuda_memory_eval PRIVATE
    GELU_CUDA_COMPILER_VERSION=\"${CMAKE_CUDA_COMPILER_VERSION}\")
set_target_properties(gelu_cuda_memory_eval PROPERTIES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF)
add_test(NAME gelu_cuda_memory_eval
    COMMAND gelu_cuda_memory_eval "${GELU_CUDA_MEMORY_RESULTS}" 65536)
set_tests_properties(gelu_cuda_memory_eval PROPERTIES SKIP_RETURN_CODE 77)

option(OPS_BUILD_PERFORMANCE "Build the CUDA performance evaluation executable" OFF)
if(OPS_BUILD_PERFORMANCE)
    add_executable(gelu_cuda_benchmark "${OPS_TEST_PERFORMANCE_SOURCE}")
    target_include_directories(gelu_cuda_benchmark PRIVATE "${GELU_OPERATOR_ROOT}/src/cuda/include")
    target_link_libraries(gelu_cuda_benchmark PRIVATE gelu_cuda CUDA::cudart)
    target_compile_definitions(gelu_cuda_benchmark PRIVATE
        GELU_CUDA_KERNEL_FILE="${OPS_KERNEL_FILE}"
        GELU_CUDA_HOST_FILE="${OPS_HOST_FILE}")
    set_target_properties(gelu_cuda_benchmark PROPERTIES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF)
endif()
