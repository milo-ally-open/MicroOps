#include "ops/div_cann.hpp"
#include "kernel_selection.h"

namespace ops {
aclError div_cann(const float* A, const float* B, float* C,
                  int32_t N, aclrtStream stream) {
    if (N < 0 || (N > 0 && (A == nullptr || B == nullptr || C == nullptr))) {
        return ACL_ERROR_INVALID_PARAM;
    }
    if (N == 0) return ACL_SUCCESS;
    if (stream == nullptr) return ACL_ERROR_INVALID_PARAM;

    // The validated board exposes one AI Core. Each v2 loop iteration handles
    // four values and the per-element guards protect a non-multiple tail.
    div_kernel_v2<<<1, nullptr, stream>>>(
        const_cast<float*>(A), const_cast<float*>(B), C,
        static_cast<uint32_t>(N));
    return ACL_SUCCESS;
}
}  // namespace ops
