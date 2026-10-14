add_library(add_cuda STATIC host/add_cuda.cu)
target_include_directories(add_cuda PUBLIC include)
target_link_libraries(add_cuda PUBLIC CUDA::cudart)
set_target_properties(add_cuda PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF)
