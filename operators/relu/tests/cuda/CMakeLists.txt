cmake_minimum_required(VERSION 3.21)
project(relu_cuda_tests VERSION 0.1.0 LANGUAGES CXX)

include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR
        "The relu CUDA test project requires nvcc/CUDA Toolkit. Install CUDA or run a backend available on this host.")
endif()
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
include(CTest)

function(ops_fix_pie_link target)
    if(UNIX AND NOT APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_link_options(${target} PRIVATE -no-pie)
    endif()
endfunction()

get_filename_component(RELU_OPERATOR_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
add_subdirectory("${RELU_OPERATOR_ROOT}/src/cuda" "${CMAKE_CURRENT_BINARY_DIR}/operator_cuda")

set(OPS_TEST_CORRECTNESS_SOURCE "${CMAKE_CURRENT_LIST_DIR}/test/relu_cuda_test.cu"
    CACHE FILEPATH "CUDA correctness test source")
set(OPS_TEST_MEMORY_SOURCE "${CMAKE_CURRENT_LIST_DIR}/test/relu_cuda_memory_eval.cu"
    CACHE FILEPATH "CUDA memory evaluation source")
set(OPS_TEST_BENCH_SOURCE "${CMAKE_CURRENT_LIST_DIR}/bench/relu_cuda_bench.cu"
    CACHE FILEPATH "CUDA bench evaluation source")

add_executable(relu_cuda_correctness_test "${OPS_TEST_CORRECTNESS_SOURCE}")
ops_fix_pie_link(relu_cuda_correctness_test)
target_link_libraries(relu_cuda_correctness_test PRIVATE relu_cuda CUDA::cudart)
target_compile_definitions(relu_cuda_correctness_test PRIVATE
    RELU_CUDA_KERNEL_FILE="${OPS_KERNEL_FILE}"
    RELU_CUDA_HOST_FILE="${OPS_HOST_FILE}")
set_target_properties(relu_cuda_correctness_test PROPERTIES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF)
add_test(NAME relu_cuda_correctness COMMAND relu_cuda_correctness_test)
set_tests_properties(relu_cuda_correctness PROPERTIES SKIP_RETURN_CODE 77)

set(RELU_CUDA_MEMORY_RESULTS "${RELU_OPERATOR_ROOT}/tests/cuda/test/results/memory.csv"
    CACHE FILEPATH "CSV path for real CUDA memory-evaluation records")
add_executable(relu_cuda_memory_eval "${OPS_TEST_MEMORY_SOURCE}")
ops_fix_pie_link(relu_cuda_memory_eval)
target_link_libraries(relu_cuda_memory_eval PRIVATE CUDA::cudart)
target_compile_definitions(relu_cuda_memory_eval PRIVATE
    RELU_CUDA_COMPILER_VERSION=\"${CMAKE_CUDA_COMPILER_VERSION}\")
set_target_properties(relu_cuda_memory_eval PROPERTIES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF)
add_test(NAME relu_cuda_memory_eval
    COMMAND relu_cuda_memory_eval "${RELU_CUDA_MEMORY_RESULTS}" 65536)
set_tests_properties(relu_cuda_memory_eval PROPERTIES SKIP_RETURN_CODE 77)

option(OPS_BUILD_BENCH "Build the CUDA bench evaluation executable" OFF)
if(OPS_BUILD_BENCH)
    add_executable(relu_cuda_bench "${OPS_TEST_BENCH_SOURCE}")
    ops_fix_pie_link(relu_cuda_bench)
    target_include_directories(relu_cuda_bench PRIVATE "${RELU_OPERATOR_ROOT}/src/cuda/include")
    target_link_libraries(relu_cuda_bench PRIVATE relu_cuda CUDA::cudart)
    target_compile_definitions(relu_cuda_bench PRIVATE
        RELU_CUDA_KERNEL_FILE="${OPS_KERNEL_FILE}"
        RELU_CUDA_HOST_FILE="${OPS_HOST_FILE}")
    set_target_properties(relu_cuda_bench PROPERTIES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF)
endif()
