// Host launcher for sigmoid_kernel_v1.
#include "ops/sigmoid_cuda.hpp"
#include "kernel_selection.h"

namespace ops {
cudaError_t sigmoid_cuda(const float* X, float* Y, const int N, const cudaStream_t stream) {
    if (N < 0 || (N > 0 && (X == nullptr || Y == nullptr))) return cudaErrorInvalidValue;
    if (N == 0) return cudaSuccess;
    constexpr int threads_per_block = 256;
    constexpr int items_per_thread = 1;
    constexpr int items_per_block = threads_per_block * items_per_thread;
    const int blocks = (N - 1) / items_per_block + 1;
    sigmoid_kernel_v1<<<blocks, threads_per_block, 0, stream>>>(X, Y, N);
    return cudaPeekAtLastError();
}
}  // namespace ops
