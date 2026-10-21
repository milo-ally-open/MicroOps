// CANN CCE scaffold: process four independent scalar elements per loop iteration.
#include "kernel_macros.h"
#include <cstdint>
#include <cmath>

extern "C" __global__ __aicore__ void gelu_kernel_v2(
    __gm__ float* X, __gm__ float* Y, uint32_t N) {
    constexpr uint32_t kElementsPerIteration = 4;
    const uint32_t first = block_idx * kElementsPerIteration;
    const uint32_t stride = block_num * kElementsPerIteration;
    for (uint32_t base = first; base < N; base += stride) {
        if (base < N) { const float x = X[base]; Y[base] = (0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)))); }
        if (base + 1 < N) { const float x = X[base + 1]; Y[base + 1] = (0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)))); }
        if (base + 2 < N) { const float x = X[base + 2]; Y[base + 2] = (0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)))); }
        if (base + 3 < N) { const float x = X[base + 3]; Y[base + 3] = (0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)))); }
    }
}
