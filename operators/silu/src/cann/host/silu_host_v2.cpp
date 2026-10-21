// Host launcher for silu_kernel_v2; scaffold pending CANN device validation.
#include "ops/silu_cann.hpp"
#include "kernel_selection.h"

namespace ops {
aclError silu_cann(const float* X, float* Y, int32_t N, aclrtStream stream) {
    if (N < 0 || (N > 0 && (X == nullptr || Y == nullptr))) return ACL_ERROR_INVALID_PARAM;
    if (N == 0) return ACL_SUCCESS;
    if (stream == nullptr) return ACL_ERROR_INVALID_PARAM;
    silu_kernel_v2<<<1, nullptr, stream>>>(
        const_cast<float*>(X), Y, static_cast<uint32_t>(N));
    return ACL_SUCCESS;
}
}  // namespace ops
