#pragma once
#include <cuda_runtime_api.h>
namespace ops {
// X and Y are device pointers. The kernel is enqueued on stream; callers synchronize before reading Y.
cudaError_t gelu_cuda(const float* X, float* Y, int N, cudaStream_t stream);
}  // namespace ops
