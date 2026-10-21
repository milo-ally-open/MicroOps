#pragma once
#include <acl/acl.h>
#include <cstdint>
namespace ops {
// X and Y are device pointers. Enqueues float32 elementwise gelu on stream; N=0 is a no-op.
aclError gelu_cann(const float* X, float* Y, int32_t N, aclrtStream stream);
}  // namespace ops
