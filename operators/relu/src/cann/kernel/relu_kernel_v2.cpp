// CANN CCE scaffold: process four independent scalar elements per loop iteration.
#include "kernel_macros.h"
#include <cstdint>

extern "C" __global__ __aicore__ void relu_kernel_v2(
    __gm__ float* X, __gm__ float* Y, uint32_t N) {
    constexpr uint32_t kElementsPerIteration = 4;
    const uint32_t first = block_idx * kElementsPerIteration;
    const uint32_t stride = block_num * kElementsPerIteration;
    for (uint32_t base = first; base < N; base += stride) {
        if (base < N) { const float x = X[base]; Y[base] = (x > 0.0f ? x : (x != x ? x : 0.0f)); }
        if (base + 1 < N) { const float x = X[base + 1]; Y[base + 1] = (x > 0.0f ? x : (x != x ? x : 0.0f)); }
        if (base + 2 < N) { const float x = X[base + 2]; Y[base + 2] = (x > 0.0f ? x : (x != x ? x : 0.0f)); }
        if (base + 3 < N) { const float x = X[base + 3]; Y[base + 3] = (x > 0.0f ? x : (x != x ? x : 0.0f)); }
    }
}
