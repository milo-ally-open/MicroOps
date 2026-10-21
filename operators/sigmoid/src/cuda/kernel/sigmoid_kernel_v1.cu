// sigmoid v1: one CUDA thread computes one float32 element.
#include <cmath>

__global__ void sigmoid_kernel_v1(const float* X, float* Y, const int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        const float x = X[idx];
        Y[idx] = (x >= 0.0f ? (1.0f / (1.0f + expf(-x))) : (expf(x) / (1.0f + expf(x))));
    }
}
