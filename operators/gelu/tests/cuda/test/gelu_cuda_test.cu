#include "ops/gelu_cuda.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <vector>
namespace {
constexpr int kSkip = 77;
float reference(float x) { return 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x))); }
bool run_case(int n, int offset = 0) {
    if (n == 0) return ops::gelu_cuda(nullptr, nullptr, 0, nullptr) == cudaSuccess;
    std::vector<float> input(static_cast<size_t>(n)), output(static_cast<size_t>(n));
    const float samples[] = {-8.0f, -3.0f, -0.5f, 0.0f, 0.5f, 3.0f, 8.0f};
    for (int i = 0; i < n; ++i) input[i] = samples[i % (sizeof(samples) / sizeof(samples[0]))];
    float *dx = nullptr, *dy = nullptr;
    const size_t bytes = static_cast<size_t>(n + offset) * sizeof(float);
    cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&dx), bytes);
    if (status == cudaSuccess) status = cudaMalloc(reinterpret_cast<void**>(&dy), bytes);
    if (status == cudaSuccess) status = cudaMemcpy(dx + offset, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    if (status == cudaSuccess) status = ops::gelu_cuda(dx + offset, dy + offset, n, nullptr);
    if (status == cudaSuccess) status = cudaDeviceSynchronize();
    if (status == cudaSuccess) status = cudaMemcpy(output.data(), dy + offset, n * sizeof(float), cudaMemcpyDeviceToHost);
    bool ok = status == cudaSuccess;
    for (int i = 0; ok && i < n; ++i) {
        const float expected = reference(input[i]);
        const float error = std::fabs(output[i] - expected);
        const float limit = 5e-5f + 5e-5f * fabsf(expected);
        if (!(error <= limit) && !(std::isnan(output[i]) && std::isnan(expected))) {
            std::cerr << "mismatch N=" << n << " i=" << i << " expected=" << expected << " got=" << output[i] << '\n'; ok = false;
        }
    }
    if (dy) cudaFree(dy); if (dx) cudaFree(dx);
    return ok;
}
}
int main() {
    if (ops::gelu_cuda(nullptr, nullptr, -1, nullptr) != cudaErrorInvalidValue ||
        ops::gelu_cuda(nullptr, nullptr, 1, nullptr) != cudaErrorInvalidValue) return 1;
    int count = 0; const cudaError_t device_status = cudaGetDeviceCount(&count);
    if (device_status == cudaErrorNoDevice || device_status == cudaErrorInsufficientDriver ||
        device_status == cudaErrorInitializationError || (device_status == cudaSuccess && count == 0)) {
        std::cout << "SKIP: CUDA device/driver unavailable\n"; return kSkip;
    }
    if (device_status != cudaSuccess) { std::cerr << cudaGetErrorString(device_status) << '\n'; return 1; }
    if (!run_case(0) || !run_case(1) || !run_case(259) || !run_case(1027) || !run_case(259, 1)) return 1;
    std::cout << "CUDA gelu correctness passed on " << count << " device(s)\n"; return 0;
}
