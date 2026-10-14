// CANN v2 candidate: four independent scalar adds per loop iteration.
// This exposes instruction-level parallelism on the validated dav-m300 target;
// it is not a claim that the compiler emits a SIMD vector instruction.
#include "kernel_macros.h"
#include <cstdint>

extern "C" __global__ __aicore__ void add_kernel_v2(
    __gm__ float* A, __gm__ float* B, __gm__ float* C, uint32_t N) {
    constexpr uint32_t kElementsPerIteration = 4;
    const uint32_t first = block_idx * kElementsPerIteration;
    const uint32_t stride = block_num * kElementsPerIteration;
    for (uint32_t base = first; base < N; base += stride) {
        if (base < N) C[base] = A[base] + B[base];
        if (base + 1 < N) C[base + 1] = A[base + 1] + B[base + 1];
        if (base + 2 < N) C[base + 2] = A[base + 2] + B[base + 2];
        if (base + 3 < N) C[base + 3] = A[base + 3] + B[base + 3];
    }
}
