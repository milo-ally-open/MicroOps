#pragma once
#include <acl/acl.h>
#include <cstdint>
namespace ops {
// X and Y are device pointers. Enqueues float32 elementwise sigmoid on stream; N=0 is a no-op.
aclError sigmoid_cann(const float* X, float* Y, int32_t N, aclrtStream stream);
}  // namespace ops
