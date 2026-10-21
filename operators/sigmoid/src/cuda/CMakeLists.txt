set(OPS_KERNEL_FILE "" CACHE STRING "Selected CUDA kernel source filename")
set(OPS_HOST_FILE "" CACHE STRING "Selected CUDA host source filename")
set(OPS_KERNEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernel" CACHE PATH "CUDA kernel source directory")
set(OPS_HOST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/host" CACHE PATH "CUDA host source directory")
if(OPS_KERNEL_FILE STREQUAL "" OR OPS_HOST_FILE STREQUAL "")
    message(FATAL_ERROR "Set both OPS_KERNEL_FILE and OPS_HOST_FILE to source filenames under kernel/ and host/.")
endif()
get_filename_component(SIGMOID_KERNEL_FILENAME "${OPS_KERNEL_FILE}" NAME)
get_filename_component(SIGMOID_HOST_FILENAME "${OPS_HOST_FILE}" NAME)
if(NOT SIGMOID_KERNEL_FILENAME STREQUAL OPS_KERNEL_FILE OR NOT SIGMOID_HOST_FILENAME STREQUAL OPS_HOST_FILE)
    message(FATAL_ERROR "OPS_KERNEL_FILE and OPS_HOST_FILE must be filenames, not paths.")
endif()
set(SIGMOID_KERNEL_SOURCE "${OPS_KERNEL_DIR}/${OPS_KERNEL_FILE}")
set(SIGMOID_HOST_SOURCE "${OPS_HOST_DIR}/${OPS_HOST_FILE}")
if(NOT EXISTS "${SIGMOID_KERNEL_SOURCE}")
    message(FATAL_ERROR "CUDA kernel source was not found: ${SIGMOID_KERNEL_SOURCE}")
endif()
if(NOT EXISTS "${SIGMOID_HOST_SOURCE}")
    message(FATAL_ERROR "CUDA host source was not found: ${SIGMOID_HOST_SOURCE}")
endif()

set(SIGMOID_KERNEL_HEADER_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${SIGMOID_KERNEL_HEADER_DIR}")
configure_file(host/kernel_selection.h.in
    "${SIGMOID_KERNEL_HEADER_DIR}/kernel_selection.h" @ONLY)

add_library(sigmoid_cuda STATIC "${SIGMOID_HOST_SOURCE}")
target_include_directories(sigmoid_cuda PUBLIC include PRIVATE "${SIGMOID_KERNEL_HEADER_DIR}" "${OPS_KERNEL_DIR}")
target_link_libraries(sigmoid_cuda PUBLIC CUDA::cudart)
set_target_properties(sigmoid_cuda PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF)
