// add v1 elementwise kernel: one thread computes one float32 element.
// Validation status and the CMake integration status are recorded in operator.yaml.

__global__ void add_kernel_v1(const float* A, const float* B, float* C, int N) {
    // 1. 计算全局线程索引
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 2. 边界保护
    if (idx < N) {
        // 3. 逐元素计算
        C[idx] = A[idx] + B[idx];
    }
}
