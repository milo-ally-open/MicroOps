#include <cuda_runtime.h>

constexpr int THREADS = 256;

__device__ __forceinline__ float4 relu4(float4 value) {
    return make_float4(
        fmaxf(value.x, 0.0f),
        fmaxf(value.y, 0.0f),
        fmaxf(value.z, 0.0f),
        fmaxf(value.w, 0.0f));
}
__global__ void relu_kernel(const float* input, float* output, int N) {
    const size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    const size_t n4 = static_cast<size_t>(N) / 4;
    const float4* input4 = reinterpret_cast<const float4*>(input);
    float4* output4 = reinterpret_cast<float4*>(output);

    for (size_t i = tid; i < n4; i += stride) {
        output4[i] = relu4(input4[i]);
    }

    const size_t tail_start = n4 * 4;
    for (size_t i = tail_start + tid; i < static_cast<size_t>(N); i += stride) {
        output[i] = fmaxf(input[i], 0.0f);
    }
}

void solve(const float* input, float* output, int N) {
    if (N <= 0) {
        return;
    }

    const int vector_count = N / 4;
    const int tail_count = N - vector_count * 4;
    const int work_items = vector_count > tail_count ? vector_count : tail_count;
    const int blocks = (work_items + THREADS - 1) / THREADS;
    relu_kernel<<<blocks, THREADS>>>(input, output, N);
}
