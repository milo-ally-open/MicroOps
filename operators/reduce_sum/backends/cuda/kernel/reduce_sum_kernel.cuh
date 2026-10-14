// SCAFFOLD_UNVERIFIED: candidate implementation; no CUDA compiler or device
// validation has been performed for this repository snapshot.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace ops::cuda_detail {

constexpr unsigned int kThreadsPerBlock = 256;
constexpr std::size_t kElementsPerThread = 4;
constexpr std::size_t kMaxPartialBlocks = 65535;

__global__ void reduce_sum_partials(const float* input,
                                    const std::size_t count,
                                    float* partials) {
    __shared__ float shared[kThreadsPerBlock];
    const unsigned int lane = threadIdx.x;
    const std::size_t first =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x * kElementsPerThread + lane;
    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x * kElementsPerThread;

    float sum = 0.0F;
    for (std::size_t base = first; base < count; base += stride) {
        for (std::size_t offset = 0; offset < kElementsPerThread; ++offset) {
            const std::size_t index = base + offset * blockDim.x;
            if (index < count) {
                sum += input[index];
            }
        }
    }
    shared[lane] = sum;
    __syncthreads();

    for (unsigned int width = blockDim.x / 2; width > 0; width >>= 1) {
        if (lane < width) {
            shared[lane] += shared[lane + width];
        }
        __syncthreads();
    }
    if (lane == 0) {
        partials[blockIdx.x] = shared[0];
    }
}

__global__ void finish_reduce_sum(const float* partials,
                                  const std::size_t partial_count,
                                  float* output) {
    __shared__ float shared[kThreadsPerBlock];
    const unsigned int lane = threadIdx.x;
    float sum = 0.0F;
    for (std::size_t index = lane; index < partial_count; index += blockDim.x) {
        sum += partials[index];
    }
    shared[lane] = sum;
    __syncthreads();
    for (unsigned int width = blockDim.x / 2; width > 0; width >>= 1) {
        if (lane < width) {
            shared[lane] += shared[lane + width];
        }
        __syncthreads();
    }
    if (lane == 0) {
        *output = shared[0];
    }
}

}  // namespace ops::cuda_detail
\n