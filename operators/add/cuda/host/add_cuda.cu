// Host launch wrapper for add v1. See operator.yaml for current validation status.
#include "ops/add_cuda.hpp"

#include "../kernel/add_kernel.cu"

namespace ops {

cudaError_t add_cuda_v1(const float* A,
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
    const int blocks = (N - 1) / threads_per_block + 1;
    add_kernel_v1<<<blocks, threads_per_block, 0, stream>>>(A, B, C, N);
    return cudaPeekAtLastError();
}

}  // namespace ops
