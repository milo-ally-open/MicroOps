cmake_minimum_required(VERSION 3.21)
project(OpsFromScratch VERSION 0.1.0 LANGUAGES CXX)

include(cmake/ProjectOptions.cmake)

option(OPS_ENABLE_CUDA "Build the CUDA backend" OFF)
option(OPS_ENABLE_ASCENDC "Build the Ascend C backend" OFF)
option(OPS_BUILD_TESTS "Build CPU reference tests" ON)
option(OPS_BUILD_EXAMPLES "Build operator examples" OFF)
option(OPS_BUILD_BENCHMARKS "Build benchmark executables" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(reduce_sum_reference STATIC
    operators/reduce_sum/reference/src/reduce_sum.cpp)
target_include_directories(reduce_sum_reference PUBLIC
    operators/reduce_sum/reference/include)
target_compile_features(reduce_sum_reference PUBLIC cxx_std_17)

if(OPS_ENABLE_CUDA)
    add_subdirectory(operators/reduce_sum/backends/cuda)
endif()

if(OPS_ENABLE_ASCENDC)
    add_subdirectory(operators/reduce_sum/backends/ascendc)
endif()

if(OPS_BUILD_TESTS)
    include(CTest)
    add_executable(reduce_sum_reference_test
        operators/reduce_sum/tests/reduce_sum_reference_test.cpp)
    target_link_libraries(reduce_sum_reference_test PRIVATE reduce_sum_reference)
    add_test(NAME reduce_sum_reference COMMAND reduce_sum_reference_test)
endif()

if(OPS_BUILD_EXAMPLES)
    add_executable(reduce_sum_grayscale_mean
        operators/reduce_sum/examples/grayscale_mean.cpp)
    target_link_libraries(reduce_sum_grayscale_mean PRIVATE reduce_sum_reference)
endif()

if(OPS_BUILD_BENCHMARKS)
    add_executable(reduce_sum_cpu_benchmark
        operators/reduce_sum/benchmarks/cpu_benchmark.cpp)
    target_link_libraries(reduce_sum_cpu_benchmark PRIVATE reduce_sum_reference)
endif()

\n