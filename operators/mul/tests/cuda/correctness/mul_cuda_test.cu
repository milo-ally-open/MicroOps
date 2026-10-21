#include "ops/mul_cuda.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <vector>

namespace {

constexpr int kSkip = 77;

bool check_cuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return true;
    }
    std::cerr << operation << " failed: " << cudaGetErrorString(status) << '\n';
    return false;
}

bool run_case(const int count, const int device_offset = 0) {
    if (count == 0) {
        return check_cuda(ops::mul_cuda(nullptr, nullptr, nullptr, 0, nullptr),
                          "zero-length mul_cuda");
    }

    std::vector<float> host_a(static_cast<std::size_t>(count));
    std::vector<float> host_b(static_cast<std::size_t>(count));
    std::vector<float> host_c(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        host_a[static_cast<std::size_t>(i)] = static_cast<float>((i % 37) - 18) * 0.5f;
        host_b[static_cast<std::size_t>(i)] = static_cast<float>(i % 11) - 5.0f;
    }

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_c = nullptr;
    const std::size_t allocation_bytes =
        static_cast<std::size_t>(count + device_offset) * sizeof(float);
    cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&device_a), allocation_bytes);
    if (status == cudaSuccess) {
        status = cudaMalloc(reinterpret_cast<void**>(&device_b), allocation_bytes);
    }
    if (status == cudaSuccess) {
        status = cudaMalloc(reinterpret_cast<void**>(&device_c), allocation_bytes);
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(device_a + device_offset, host_a.data(),
                            count * sizeof(float), cudaMemcpyHostToDevice);
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(device_b + device_offset, host_b.data(),
                            count * sizeof(float), cudaMemcpyHostToDevice);
    }
    if (status == cudaSuccess) {
        status = ops::mul_cuda(device_a + device_offset, device_b + device_offset,
                               device_c + device_offset, count, nullptr);
    }
    if (status == cudaSuccess) {
        status = cudaDeviceSynchronize();
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(host_c.data(), device_c + device_offset,
                            count * sizeof(float), cudaMemcpyDeviceToHost);
    }

    bool valid = check_cuda(status, "CUDA mul test operation");
    if (valid) {
        for (int i = 0; i < count; ++i) {
            const auto index = static_cast<std::size_t>(i);
            const float expected = host_a[index] * host_b[index];
            if (std::fabs(host_c[index] - expected) > 1e-6f) {
                std::cerr << "mismatch at index " << i << ": expected " << expected
                          << ", got " << host_c[index] << '\n';
                valid = false;
                break;
            }
        }
    }

    if (device_c != nullptr) cudaFree(device_c);
    if (device_b != nullptr) cudaFree(device_b);
    if (device_a != nullptr) cudaFree(device_a);
    return valid;
}

}  // namespace

int main() {
    if (ops::mul_cuda(nullptr, nullptr, nullptr, -1, nullptr) != cudaErrorInvalidValue ||
        ops::mul_cuda(nullptr, nullptr, nullptr, 1, nullptr) != cudaErrorInvalidValue) {
        std::cerr << "invalid size/pointers must return cudaErrorInvalidValue\n";
        return 1;
    }
    int device_count = 0;
    const cudaError_t device_status = cudaGetDeviceCount(&device_count);
    if (device_status == cudaErrorNoDevice ||
        device_status == cudaErrorInsufficientDriver ||
        device_status == cudaErrorInitializationError ||
        (device_status == cudaSuccess && device_count == 0)) {
        std::cout << "SKIP: CUDA device/driver unavailable";
        if (device_status != cudaSuccess) {
            std::cout << " (" << cudaGetErrorString(device_status) << ')';
        }
        std::cout << '\n';
        return kSkip;
    }
    if (!check_cuda(device_status, "cudaGetDeviceCount")) {
        return 1;
    }

    if (!run_case(0) || !run_case(1) || !run_case(259) || !run_case(259, 1)) {
        return 1;
    }
    std::cout << "CUDA mul kernel=" << MUL_CUDA_KERNEL_FILE
              << " host=" << MUL_CUDA_HOST_FILE
              << " tests passed on " << device_count << " device(s)\n";
    return 0;
}
