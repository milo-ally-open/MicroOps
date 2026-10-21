#pragma once

#include <acl/acl.h>
#include <cstdint>

namespace ops {
// A, B and C are device pointers. Enqueues float32 elementwise subtraction.
// The caller owns allocation and synchronization. N=0 is a no-op.
aclError sub_cann(const float* A, const float* B, float* C,
                  int32_t N, aclrtStream stream);
}  // namespace ops
