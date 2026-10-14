// SCAFFOLD_UNVERIFIED: host wrapper; requires validation with the supported
// CUDA Toolkit version and an actual CUDA device before claiming availability.
#include "ops/reduce_sum_cuda.hpp"

#include "../kernel/reduce_sum_kernel.cuh"

#include <algorithm>
#include <cstddef>

namespace ops {

cudaError_t reduce_sum_cuda(const float* device_input,
                            const std::size_t count,
                            float* device_output,
                            const cudaStream_t stream) {
    if (device_output == nullptr || (count != 0 && device_input == nullptr)) {
        return cudaErrorInvalidValue;
    }

    constexpr std::size_t elements_per_block =
        cuda_detail::kThreadsPerBlock * cuda_detail::kElementsPerThread;
    const std::size_t requested_blocks =
        count / elements_per_block + (count % elements_per_block != 0 ? 1 : 0);
    const std::size_t block_count = count == 0
        ? 0
        : std::min(requested_blocks, cuda_detail::kMaxPartialBlocks);

    float* partials = nullptr;
    if (block_count != 0) {
        const cudaError_t allocation =
            cudaMalloc(reinterpret_cast<void**>(&partials), block_count * sizeof(float));
        if (allocation != cudaSuccess) {
            return allocation;
        }
        cuda_detail::reduce_sum_partials<<<static_cast<unsigned int>(block_count),
                                           cuda_detail::kThreadsPerBlock,
                                           0,
                                           stream>>>(device_input, count, partials);
        const cudaError_t partial_launch = cudaPeekAtLastError();
        if (partial_launch != cudaSuccess) {
            (void)cudaFree(partials);
            return partial_launch;
        }
    }

    cuda_detail::finish_reduce_sum<<<1, cuda_detail::kThreadsPerBlock, 0, stream>>>(
        partials, block_count, device_output);
    const cudaError_t finish_launch = cudaPeekAtLastError();
    if (partials != nullptr) {
        const cudaError_t release = cudaFree(partials);
        if (finish_launch != cudaSuccess) {
            return finish_launch;
        }
        if (release != cudaSuccess) {
            return release;
        }
    }
    return finish_launch;
}

}  // namespace ops
