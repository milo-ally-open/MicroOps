// mul v2: vectorized float4 path for aligned inputs, with a scalar fallback.
// Each thread handles up to four contiguous elements; the last thread handles
// the tail. Unaligned pointers use a grid-stride scalar loop.

#include <cstdint>

__global__ void mul_kernel_v2(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              const int N) {
    const int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int thread_stride = blockDim.x * gridDim.x;
    const std::uintptr_t addresses = reinterpret_cast<std::uintptr_t>(A) |
                                     reinterpret_cast<std::uintptr_t>(B) |
                                     reinterpret_cast<std::uintptr_t>(C);
    constexpr std::uintptr_t kFloat4AlignmentMask = alignof(float4) - 1;
    const bool aligned = (addresses & kFloat4AlignmentMask) == 0;

    if (!aligned) {
        for (int index = thread_index; index < N; index += thread_stride) {
            C[index] = A[index] * B[index];
        }
        return;
    }

    const int vector_count = N / 4;
    for (int vector_index = thread_index;
         vector_index < vector_count;
         vector_index += thread_stride) {
        const float4 a = reinterpret_cast<const float4*>(A)[vector_index];
        const float4 b = reinterpret_cast<const float4*>(B)[vector_index];
        float4 c{};
        c.x = a.x * b.x;
        c.y = a.y * b.y;
        c.z = a.z * b.z;
        c.w = a.w * b.w;
        reinterpret_cast<float4*>(C)[vector_index] = c;
    }

    if (thread_index == 0) {
        for (int index = vector_count * 4; index < N; ++index) {
            C[index] = A[index] * B[index];
        }
    }
}
