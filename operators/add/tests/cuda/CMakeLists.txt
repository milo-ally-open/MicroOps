cmake_minimum_required(VERSION 3.21)
project(add_cuda_tests VERSION 0.1.0 LANGUAGES CXX)

include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR
        "The add CUDA test project requires nvcc/CUDA Toolkit. Install CUDA or run a backend available on this host.")
endif()
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
include(CTest)

get_filename_component(ADD_OPERATOR_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
add_subdirectory("${ADD_OPERATOR_ROOT}/cuda" "${CMAKE_CURRENT_BINARY_DIR}/operator_cuda")

add_executable(add_cuda_correctness_test correctness/add_cuda_test.cu)
target_link_libraries(add_cuda_correctness_test PRIVATE add_cuda CUDA::cudart)
set_target_properties(add_cuda_correctness_test PROPERTIES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF)
add_test(NAME add_cuda_correctness COMMAND add_cuda_correctness_test)
set_tests_properties(add_cuda_correctness PROPERTIES SKIP_RETURN_CODE 77)

set(ADD_CUDA_MEMORY_RESULTS
    "${ADD_OPERATOR_ROOT}/benchmarks/cuda/results/memory.csv"
    CACHE FILEPATH "CSV path for real CUDA memory-evaluation records")
add_executable(add_cuda_memory_eval memory/add_cuda_memory_eval.cu)
target_link_libraries(add_cuda_memory_eval PRIVATE CUDA::cudart)
target_compile_definitions(add_cuda_memory_eval PRIVATE
    ADD_CUDA_COMPILER_VERSION=\"${CMAKE_CUDA_COMPILER_VERSION}\")
set_target_properties(add_cuda_memory_eval PROPERTIES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF)
add_test(NAME add_cuda_memory_eval
    COMMAND add_cuda_memory_eval "${ADD_CUDA_MEMORY_RESULTS}" 65536)
set_tests_properties(add_cuda_memory_eval PROPERTIES SKIP_RETURN_CODE 77)

option(OPS_BUILD_PERFORMANCE "Build the CUDA performance evaluation executable" OFF)
if(OPS_BUILD_PERFORMANCE)
    add_executable(add_cuda_benchmark performance/add_cuda_benchmark.cu)
    target_include_directories(add_cuda_benchmark PRIVATE "${ADD_OPERATOR_ROOT}/cuda/include")
    target_link_libraries(add_cuda_benchmark PRIVATE add_cuda CUDA::cudart)
    set_target_properties(add_cuda_benchmark PROPERTIES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF)
endif()
