// Select one-element or float4 processing based on the measured input range.
#include "ops/mul_cuda.hpp"
#include "kernel_selection.h"

namespace ops {
cudaError_t mul_cuda(const float* A, const float* B, float* C,
                     const int N, const cudaStream_t stream) {
    if (N < 0 || (N > 0 && (A == nullptr || B == nullptr || C == nullptr))) {
        return cudaErrorInvalidValue;
    }
    if (N == 0) return cudaSuccess;

    constexpr int kThreadsPerBlock = 256;
    constexpr int kVectorizedThreshold = 262144;
    if (N < kVectorizedThreshold) {
        const int blocks = (N - 1) / kThreadsPerBlock + 1;
        mul_kernel_v1<<<blocks, kThreadsPerBlock, 0, stream>>>(A, B, C, N);
    } else {
        constexpr int kItemsPerThread = 4;
        constexpr int kItemsPerBlock = kThreadsPerBlock * kItemsPerThread;
        const int blocks = (N - 1) / kItemsPerBlock + 1;
        mul_kernel_v2<<<blocks, kThreadsPerBlock, 0, stream>>>(A, B, C, N);
    }
    return cudaPeekAtLastError();
}
}  // namespace ops
