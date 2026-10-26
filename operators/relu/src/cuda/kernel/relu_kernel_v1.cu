// relu v1: one CUDA thread computes one float32 element.

__global__ void relu_kernel_v1(const float* X, float* Y, const int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        const float x = X[idx];
        // Ordered comparison is false for NaN, so this preserves NaN without
        // a second is-NaN comparison. It also preserves the sign of -0.0f.
        Y[idx] = (x < 0.0f ? 0.0f : x);
    }
}
