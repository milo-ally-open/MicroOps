cmake_minimum_required(VERSION 3.21)
project(relu_cann LANGUAGES CXX)

set(OPS_KERNEL_FILE "" CACHE STRING "Selected CANN kernel source filename")
set(OPS_HOST_FILE "" CACHE STRING "Selected CANN host source filename")
set(OPS_KERNEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernel" CACHE PATH "CANN kernel source directory")
set(OPS_HOST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/host" CACHE PATH "CANN host source directory")
if(OPS_KERNEL_FILE STREQUAL "" OR OPS_HOST_FILE STREQUAL "")
    message(FATAL_ERROR "Set OPS_KERNEL_FILE and OPS_HOST_FILE to filenames from operator.yaml.")
endif()
get_filename_component(_kernel_name "${OPS_KERNEL_FILE}" NAME)
get_filename_component(_host_name "${OPS_HOST_FILE}" NAME)
if(NOT _kernel_name STREQUAL OPS_KERNEL_FILE OR NOT _host_name STREQUAL OPS_HOST_FILE)
    message(FATAL_ERROR "Kernel and host selections must be filenames, not paths.")
endif()
set(RELU_KERNEL_SOURCE "${OPS_KERNEL_DIR}/${OPS_KERNEL_FILE}")
set(RELU_HOST_SOURCE "${OPS_HOST_DIR}/${OPS_HOST_FILE}")
foreach(_source IN ITEMS "${RELU_KERNEL_SOURCE}" "${RELU_HOST_SOURCE}")
    if(NOT EXISTS "${_source}")
        message(FATAL_ERROR "Selected CANN source does not exist: ${_source}")
    endif()
endforeach()

set(ASCEND_TOOLKIT_ROOT "$ENV{ASCEND_HOME_PATH}" CACHE PATH "CANN toolkit root")
if(NOT ASCEND_TOOLKIT_ROOT)
    set(ASCEND_TOOLKIT_ROOT "$ENV{ASCEND_TOOLKIT_HOME}")
endif()
if(NOT ASCEND_TOOLKIT_ROOT)
    set(ASCEND_TOOLKIT_ROOT "/usr/local/Ascend/ascend-toolkit/latest")
endif()
find_program(BISHENG_COMPILER NAMES bisheng HINTS "${ASCEND_TOOLKIT_ROOT}/compiler/ccec_compiler/bin")
if(NOT BISHENG_COMPILER)
    message(FATAL_ERROR "CANN build requires bisheng. Source <CANN>/set_env.sh and install a CANN toolkit.")
endif()
set(ASCEND_INCLUDE_DIR "${ASCEND_TOOLKIT_ROOT}/aarch64-linux/include")
set(ASCEND_LIB_DIR "${ASCEND_TOOLKIT_ROOT}/aarch64-linux/lib64")
set(ASCEND_KERNEL_INCLUDE_DIR "${ASCEND_TOOLKIT_ROOT}/compiler/tikcpp/tikcfw/impl")
if(NOT EXISTS "${ASCEND_INCLUDE_DIR}/acl/acl.h" OR NOT EXISTS "${ASCEND_LIB_DIR}/libascendcl.so")
    message(FATAL_ERROR "CANN ACL headers/libraries not found under ${ASCEND_TOOLKIT_ROOT}; source set_env.sh and check toolkit install.")
endif()
set(RELU_CANN_RUNTIME_VERSION "unknown" CACHE INTERNAL "Detected CANN runtime version" FORCE)
if(EXISTS "${ASCEND_TOOLKIT_ROOT}/version.cfg")
    file(STRINGS "${ASCEND_TOOLKIT_ROOT}/version.cfg" _runtime_version REGEX "^runtime_running_version=")
    string(FIND "${_runtime_version}" ":" _version_separator)
    if(_version_separator GREATER -1)
        math(EXPR _version_start "${_version_separator} + 1")
        string(LENGTH "${_runtime_version}" _version_length)
        math(EXPR _version_count "${_version_length} - ${_version_start} - 1")
        string(SUBSTRING "${_runtime_version}" ${_version_start} ${_version_count} _detected_runtime_version)
        set(RELU_CANN_RUNTIME_VERSION "${_detected_runtime_version}" CACHE INTERNAL "Detected CANN runtime version" FORCE)
    endif()
endif()

set(RELU_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${RELU_GENERATED_DIR}")
configure_file(host/kernel_selection.h.in "${RELU_GENERATED_DIR}/kernel_selection.h" @ONLY)
set(RELU_CANN_LIBRARY "${CMAKE_CURRENT_BINARY_DIR}/librelu_cann.so")
add_custom_command(OUTPUT "${RELU_CANN_LIBRARY}"
    COMMAND "${BISHENG_COMPILER}" -x cce -O2 --cce-aicore-arch=dav-m300 -std=c++17 -fPIC -shared
        "-I${ASCEND_KERNEL_INCLUDE_DIR}" "-I${ASCEND_INCLUDE_DIR}"
        "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-I${RELU_GENERATED_DIR}"
        "-L${ASCEND_LIB_DIR}" "-Wl,-rpath,${ASCEND_LIB_DIR}"
        "${RELU_HOST_SOURCE}" -lascendcl -lruntime -o "${RELU_CANN_LIBRARY}"
    DEPENDS "${RELU_HOST_SOURCE}" "${RELU_KERNEL_SOURCE}" "${RELU_GENERATED_DIR}/kernel_selection.h"
    COMMENT "Building selected CANN kernel/host with bisheng (dav-m300)"
    VERBATIM)
add_custom_target(relu_cann ALL DEPENDS "${RELU_CANN_LIBRARY}")

option(OPS_BUILD_EXAMPLES "Build configured CANN examples" OFF)
if(OPS_BUILD_EXAMPLES)
    set(RELU_CANN_DEMO "${CMAKE_CURRENT_BINARY_DIR}/relu_demo")
    add_custom_command(OUTPUT "${RELU_CANN_DEMO}"
        COMMAND "${CMAKE_CXX_COMPILER}" -std=c++17
            "-I${CMAKE_CURRENT_SOURCE_DIR}/include" "-I${ASCEND_INCLUDE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/../../examples/cann/relu_demo.cpp" "${RELU_CANN_LIBRARY}"
            "${ASCEND_LIB_DIR}/libascendcl.so" "-Wl,-rpath,${ASCEND_LIB_DIR}"
            "-Wl,-rpath,${CMAKE_CURRENT_BINARY_DIR}" -o "${RELU_CANN_DEMO}"
        DEPENDS "${RELU_CANN_LIBRARY}" "${CMAKE_CURRENT_SOURCE_DIR}/../../examples/cann/relu_demo.cpp"
        COMMENT "Building CANN relu demo" VERBATIM)
    add_custom_target(relu_cann_examples ALL DEPENDS "${RELU_CANN_DEMO}")
endif()
