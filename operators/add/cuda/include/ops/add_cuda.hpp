#pragma once

#include <cuda_runtime_api.h>

namespace ops {

// A, B and C are device pointers. The kernel is enqueued on stream; callers
// synchronize before reading C. N is limited to INT_MAX in this v1 interface.
cudaError_t add_cuda_v1(const float* A,
                        const float* B,
                        float* C,
                        int N,
                        cudaStream_t stream);

}  // namespace ops
