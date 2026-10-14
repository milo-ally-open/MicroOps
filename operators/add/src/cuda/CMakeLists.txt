set(OPS_KERNEL_FILE "" CACHE STRING "Selected CUDA kernel source filename")
set(OPS_HOST_FILE "" CACHE STRING "Selected CUDA host source filename")
set(OPS_KERNEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernel" CACHE PATH "CUDA kernel source directory")
set(OPS_HOST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/host" CACHE PATH "CUDA host source directory")
if(OPS_KERNEL_FILE STREQUAL "" OR OPS_HOST_FILE STREQUAL "")
    message(FATAL_ERROR "Set both OPS_KERNEL_FILE and OPS_HOST_FILE to source filenames under kernel/ and host/.")
endif()
get_filename_component(ADD_KERNEL_FILENAME "${OPS_KERNEL_FILE}" NAME)
get_filename_component(ADD_HOST_FILENAME "${OPS_HOST_FILE}" NAME)
if(NOT ADD_KERNEL_FILENAME STREQUAL OPS_KERNEL_FILE OR NOT ADD_HOST_FILENAME STREQUAL OPS_HOST_FILE)
    message(FATAL_ERROR "OPS_KERNEL_FILE and OPS_HOST_FILE must be filenames, not paths.")
endif()
set(ADD_KERNEL_SOURCE "${OPS_KERNEL_DIR}/${OPS_KERNEL_FILE}")
set(ADD_HOST_SOURCE "${OPS_HOST_DIR}/${OPS_HOST_FILE}")
if(NOT EXISTS "${ADD_KERNEL_SOURCE}")
    message(FATAL_ERROR "CUDA kernel source was not found: ${ADD_KERNEL_SOURCE}")
endif()
if(NOT EXISTS "${ADD_HOST_SOURCE}")
    message(FATAL_ERROR "CUDA host source was not found: ${ADD_HOST_SOURCE}")
endif()

set(ADD_KERNEL_HEADER_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${ADD_KERNEL_HEADER_DIR}")
configure_file(host/kernel_selection.h.in
    "${ADD_KERNEL_HEADER_DIR}/kernel_selection.h" @ONLY)

add_library(add_cuda STATIC "${ADD_HOST_SOURCE}")
target_include_directories(add_cuda PUBLIC include PRIVATE "${ADD_KERNEL_HEADER_DIR}" "${OPS_KERNEL_DIR}")
target_link_libraries(add_cuda PUBLIC CUDA::cudart)
set_target_properties(add_cuda PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF)
