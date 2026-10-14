#include "ops/reduce_sum.hpp"

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool parsePositive(const char* text, std::size_t& value) {
    if (text == nullptr || *text == '\0' || *text == '-') {
        return false;
    }
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0' || parsed == 0) {
        return false;
    }
    value = static_cast<std::size_t>(parsed);
    return static_cast<unsigned long long>(value) == parsed;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: reduce_sum_cpu_benchmark <input_size> <iterations>\n";
        return 2;
    }
    std::size_t input_size = 0;
    std::size_t iterations = 0;
    if (!parsePositive(argv[1], input_size) || !parsePositive(argv[2], iterations)) {
        std::cerr << "input_size and iterations must be positive integers\n";
        return 2;
    }

    const std::vector<float> input(input_size, 1.0F);
    for (int warmup = 0; warmup < 3; ++warmup) {
        (void)ops::reduce_sum(input.data(), input.size());
    }

    const auto begin = std::chrono::steady_clock::now();
    float sink = 0.0F;
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
        sink += ops::reduce_sum(input.data(), input.size());
    }
    const auto end = std::chrono::steady_clock::now();
    const double elapsed_us =
        std::chrono::duration<double, std::micro>(end - begin).count();
    const double average_us = elapsed_us / static_cast<double>(iterations);
    if (sink == 0.0F) {
        std::cerr << "benchmark result sanity check failed\n";
        return 1;
    }

    std::cout << input_size << ',' << iterations << ','
              << std::fixed << std::setprecision(6) << average_us << '\n';
    return 0;
}
