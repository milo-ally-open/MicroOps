#pragma once

#include <cstddef>

namespace ops {

// Sum a contiguous float32 array. Empty input returns 0.0f. A null pointer
// with a non-zero element count throws std::invalid_argument.
float reduce_sum(const float* input, std::size_t count);

}  // namespace ops
