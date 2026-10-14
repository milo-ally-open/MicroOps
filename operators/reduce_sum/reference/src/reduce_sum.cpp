#include "ops/reduce_sum.hpp"

#include <stdexcept>

namespace ops {

float reduce_sum(const float* input, const std::size_t count) {
    if (count == 0) {
        return 0.0F;
    }
    if (input == nullptr) {
        throw std::invalid_argument("reduce_sum: input is null for non-empty input");
    }

    double accumulator = 0.0;
    for (std::size_t index = 0; index < count; ++index) {
        accumulator += static_cast<double>(input[index]);
    }
    return static_cast<float>(accumulator);
}

}  // namespace ops
\n