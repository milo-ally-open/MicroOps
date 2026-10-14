// CANN 7.0 RC1-compatible scalar CCE kernel.
// The board has one AI Core; this grid-stride loop also handles arbitrary tails.
#include "kernel_macros.h"
#include <cstdint>

extern "C" __global__ __aicore__ void add_kernel_v1(
    __gm__ float* A, __gm__ float* B, __gm__ float* C, uint32_t N) {
    for (uint32_t idx = block_idx; idx < N; idx += block_num) {
        C[idx] = A[idx] + B[idx];
    }
}
