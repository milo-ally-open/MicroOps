// gelu v2: grid-stride loop with four independent scalar elements per thread.
#include <cmath>

__global__ void gelu_kernel_v2(const float* __restrict__ X,
                               float* __restrict__ Y,
                               const int N) {
    const int first = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    const int stride = blockDim.x * gridDim.x * 4;
    for (int base = first; base < N; base += stride) {
        #pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int idx = base + item * blockDim.x * gridDim.x;
            if (idx < N) {
                const float x = X[idx];
                Y[idx] = (0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x))));
            }
        }
    }
}
