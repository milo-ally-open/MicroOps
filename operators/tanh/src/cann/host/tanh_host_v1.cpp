// Host launcher for tanh_kernel_v1; scaffold pending CANN device validation.
#include "ops/tanh_cann.hpp"
#include "kernel_selection.h"

namespace ops {
aclError tanh_cann(const float* X, float* Y, int32_t N, aclrtStream stream) {
    if (N < 0 || (N > 0 && (X == nullptr || Y == nullptr))) return ACL_ERROR_INVALID_PARAM;
    if (N == 0) return ACL_SUCCESS;
    if (stream == nullptr) return ACL_ERROR_INVALID_PARAM;
    tanh_kernel_v1<<<1, nullptr, stream>>>(
        const_cast<float*>(X), Y, static_cast<uint32_t>(N));
    return ACL_SUCCESS;
}
}  // namespace ops
