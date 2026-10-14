#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace ops {

// Device pointers are expected. The operation is enqueued on stream; callers
// synchronize before reading device_output. This API is still unverified.
cudaError_t reduce_sum_cuda(const float* device_input,
                            std::size_t count,
                            float* device_output,
                            cudaStream_t stream);

}  // namespace ops
\n