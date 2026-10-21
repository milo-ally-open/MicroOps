// gelu v1: one CUDA thread computes one float32 element.
#include <cmath>

__global__ void gelu_kernel_v1(const float* X, float* Y, const int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        const float x = X[idx];
        Y[idx] = (0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x))));
    }
}
