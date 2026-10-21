#include "ops/sub_cann.hpp"
#include "kernel_selection.h"

namespace ops {
aclError sub_cann(const float* A, const float* B, float* C,
                  int32_t N, aclrtStream stream) {
    if (N < 0 || (N > 0 && (A == nullptr || B == nullptr || C == nullptr))) {
        return ACL_ERROR_INVALID_PARAM;
    }
    if (N == 0) return ACL_SUCCESS;
    if (stream == nullptr) return ACL_ERROR_INVALID_PARAM;
    // One block matches the single-AI-Core board used for initial validation.
    // The kernel grid-stride loop handles arbitrary input lengths.
    sub_kernel_v1<<<1, nullptr, stream>>>(
        const_cast<float*>(A), const_cast<float*>(B), C,
        static_cast<uint32_t>(N));
    return ACL_SUCCESS;
}
}  // namespace ops
