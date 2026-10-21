// CANN CCE scaffold; compile and validate on the target CANN toolchain/device.
#include "kernel_macros.h"
#include <cstdint>
#include <cmath>

extern "C" __global__ __aicore__ void gelu_kernel_v1(
    __gm__ float* X, __gm__ float* Y, uint32_t N) {
    for (uint32_t idx = block_idx; idx < N; idx += block_num) {
        const float x = X[idx];
        Y[idx] = (0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x))));
    }
}
