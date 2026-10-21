// Host launcher for the mul_kernel_v2 implementation.
#include "ops/mul_cuda.hpp"

#include "kernel_selection.h"

namespace ops {

cudaError_t mul_cuda(const float* A,
                     const float* B,
                     float* C,
                     const int N,
                     const cudaStream_t stream) {
    if (N < 0 || (N > 0 && (A == nullptr || B == nullptr || C == nullptr))) {
        return cudaErrorInvalidValue;
    }
    if (N == 0) {
        return cudaSuccess;
    }

    constexpr int threads_per_block = 256;
    constexpr int items_per_thread = 4;
    constexpr int items_per_block = threads_per_block * items_per_thread;
    const int blocks = (N - 1) / items_per_block + 1;
    mul_kernel_v2<<<blocks, threads_per_block, 0, stream>>>(A, B, C, N);
    return cudaPeekAtLastError();
}

}  // namespace ops
