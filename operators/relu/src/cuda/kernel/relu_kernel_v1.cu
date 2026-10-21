// relu v1: one CUDA thread computes one float32 element.

__global__ void relu_kernel_v1(const float* X, float* Y, const int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        const float x = X[idx];
        Y[idx] = (x > 0.0f ? x : (x != x ? x : 0.0f));
    }
}
