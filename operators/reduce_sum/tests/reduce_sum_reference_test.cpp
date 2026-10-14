#include "ops/reduce_sum.hpp"

#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expectNear(const std::string& name, const float actual, const float expected) {
    const float tolerance = 1.0e-5F + 1.0e-5F * std::fabs(expected);
    if (!std::isfinite(actual) || std::fabs(actual - expected) > tolerance) {
        std::cerr << "FAIL " << name << ": expected " << expected
                  << ", got " << actual << " (tolerance " << tolerance << ")\n";
        ++failures;
    } else {
        std::cout << "PASS " << name << '\n';
    }
}

void expectThrowsInvalidArgument(const std::string& name) {
    try {
        (void)ops::reduce_sum(nullptr, 1);
        std::cerr << "FAIL " << name << ": expected std::invalid_argument\n";
        ++failures;
    } catch (const std::invalid_argument&) {
        std::cout << "PASS " << name << '\n';
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << name << ": unexpected exception: " << error.what() << '\n';
        ++failures;
    }
}

}  // namespace

int main() {
    expectNear("empty input", ops::reduce_sum(nullptr, 0), 0.0F);
    const float singleton[] = {3.25F};
    expectNear("single element", ops::reduce_sum(singleton, 1), 3.25F);

    const float mixed[] = {1.0F, -2.0F, 3.5F, 0.25F};
    expectNear("mixed positive and negative", ops::reduce_sum(mixed, 4), 2.75F);
    expectThrowsInvalidArgument("null non-empty input");

    std::vector<float> large(1'000'000, 1.0F);
    expectNear("large input", ops::reduce_sum(large.data(), large.size()), 1'000'000.0F);

    const float infinityInput[] = {std::numeric_limits<float>::infinity(), 1.0F};
    if (ops::reduce_sum(infinityInput, 2) == std::numeric_limits<float>::infinity()) {
        std::cout << "PASS positive infinity propagation\n";
    } else {
        std::cerr << "FAIL positive infinity propagation\n";
        ++failures;
    }

    if (failures != 0) {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }
    std::cout << "All CPU reference checks passed\n";
    return 0;
}
\n