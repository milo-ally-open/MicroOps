// CANN CCE scaffold; compile and validate on the target CANN toolchain/device.
#include "kernel_macros.h"
#include <cstdint>
#include <cmath>

extern "C" __global__ __aicore__ void sigmoid_kernel_v1(
    __gm__ float* X, __gm__ float* Y, uint32_t N) {
    for (uint32_t idx = block_idx; idx < N; idx += block_num) {
        const float x = X[idx];
        Y[idx] = (x >= 0.0f ? (1.0f / (1.0f + expf(-x))) : (expf(x) / (1.0f + expf(x))));
    }
}
