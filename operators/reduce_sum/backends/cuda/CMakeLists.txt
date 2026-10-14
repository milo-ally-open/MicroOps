include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR
        "OPS_ENABLE_CUDA=ON, but no CUDA compiler was found. Install a compatible CUDA Toolkit/compiler, then configure this backend separately.")
endif()

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

add_library(reduce_sum_cuda STATIC
    host/reduce_sum_cuda.cu)
target_include_directories(reduce_sum_cuda PUBLIC
    include
    kernel)
target_link_libraries(reduce_sum_cuda PUBLIC CUDA::cudart)
set_target_properties(reduce_sum_cuda PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF)
