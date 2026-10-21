#include "ops/sub_cuda.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

std::string version_string(const int version) {
    return std::to_string(version / 1000) + "." +
           std::to_string((version % 1000) / 10);
}

bool parse_positive(const char* text, int* result) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value <= 0 ||
        value > std::numeric_limits<int>::max()) {
        return false;
    }
    *result = static_cast<int>(value);
    return true;
}

void print_result(const std::string& device,
                  const std::string& runtime,
                  const std::string& driver,
                  const int count,
                  const std::size_t required_bytes,
                  const std::size_t free_before,
                  const std::size_t total_bytes,
                  const std::size_t free_after_alloc,
                  const std::size_t free_after_release,
                  const std::string& allocation_status,
                  const double latency_us,
                  const double bandwidth_gbps,
                  const std::string& validation) {
    std::cout << std::setprecision(9)
              << "CUDA_RESULT\t" << SUB_CUDA_KERNEL_FILE
              << '\t' << SUB_CUDA_HOST_FILE
              << '\t' << device
              << '\t' << runtime
              << '\t' << driver
              << '\t' << count
              << '\t' << required_bytes
              << '\t' << free_before
              << '\t' << total_bytes
              << '\t' << free_after_alloc
              << '\t' << free_after_release
              << '\t' << allocation_status
              << '\t' << latency_us
              << '\t' << bandwidth_gbps
              << '\t' << validation << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: sub_cuda_benchmark <positive_N> <positive_iterations>\n";
        return 2;
    }
    int count = 0;
    int iterations = 0;
    if (!parse_positive(argv[1], &count) || !parse_positive(argv[2], &iterations)) {
        std::cerr << "N and iterations must be positive integers no larger than INT_MAX.\n";
        return 2;
    }

    int device = 0;
    cudaError_t status = cudaGetDevice(&device);
    cudaDeviceProp properties{};
    if (status == cudaSuccess) status = cudaGetDeviceProperties(&properties, device);
    int runtime_version = 0;
    int driver_version = 0;
    if (status == cudaSuccess) status = cudaRuntimeGetVersion(&runtime_version);
    if (status == cudaSuccess) status = cudaDriverGetVersion(&driver_version);
    if (status != cudaSuccess) {
        std::cerr << "CUDA device/runtime query failed: " << cudaGetErrorString(status) << '\n';
        return 1;
    }

    const std::size_t count_size = static_cast<std::size_t>(count);
    const std::size_t bytes_per_tensor = count_size * sizeof(float);
    const std::size_t working_set_bytes = 3 * bytes_per_tensor;
    std::size_t free_before = 0;
    std::size_t total_bytes = 0;
    status = cudaMemGetInfo(&free_before, &total_bytes);
    if (status != cudaSuccess) {
        std::cerr << "cudaMemGetInfo before allocation failed: "
                  << cudaGetErrorString(status) << '\n';
        return 1;
    }

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_c = nullptr;
    status = cudaMalloc(reinterpret_cast<void**>(&device_a), bytes_per_tensor);
    if (status == cudaSuccess) status = cudaMalloc(reinterpret_cast<void**>(&device_b), bytes_per_tensor);
    if (status == cudaSuccess) status = cudaMalloc(reinterpret_cast<void**>(&device_c), bytes_per_tensor);

    std::size_t free_after_alloc = 0;
    std::size_t ignored_total = 0;
    if (cudaMemGetInfo(&free_after_alloc, &ignored_total) != cudaSuccess) {
        free_after_alloc = 0;
    }
    const std::string device_name = properties.name;
    const std::string runtime = version_string(runtime_version);
    const std::string driver = version_string(driver_version);

    if (status != cudaSuccess) {
        const std::string allocation_status =
            std::string("allocation_failed:") + cudaGetErrorString(status);
        if (device_c != nullptr) cudaFree(device_c);
        if (device_b != nullptr) cudaFree(device_b);
        if (device_a != nullptr) cudaFree(device_a);
        std::size_t free_after_release = 0;
        std::size_t ignored_total = 0;
        (void)cudaMemGetInfo(&free_after_release, &ignored_total);
        print_result(device_name, runtime, driver, count, working_set_bytes,
                     free_before, total_bytes, free_after_alloc, free_after_release,
                     allocation_status, 0.0, 0.0, "not_run");
        return 0;
    }

    std::vector<float> host_a(count_size);
    std::vector<float> host_b(count_size);
    for (int i = 0; i < count; ++i) {
        host_a[static_cast<std::size_t>(i)] = static_cast<float>((i % 37) - 18) * 0.5f;
        host_b[static_cast<std::size_t>(i)] = static_cast<float>(i % 11) - 5.0f;
    }
    status = cudaMemcpy(device_a, host_a.data(), bytes_per_tensor, cudaMemcpyHostToDevice);
    if (status == cudaSuccess) status = cudaMemcpy(device_b, host_b.data(), bytes_per_tensor, cudaMemcpyHostToDevice);

    constexpr int warmup_iterations = 10;
    for (int i = 0; status == cudaSuccess && i < warmup_iterations; ++i) {
        status = ops::sub_cuda(device_a, device_b, device_c, count, nullptr);
    }
    if (status == cudaSuccess) status = cudaDeviceSynchronize();

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (status == cudaSuccess) status = cudaEventCreate(&start);
    if (status == cudaSuccess) status = cudaEventCreate(&stop);
    if (status == cudaSuccess) status = cudaEventRecord(start, nullptr);
    for (int i = 0; status == cudaSuccess && i < iterations; ++i) {
        status = ops::sub_cuda(device_a, device_b, device_c, count, nullptr);
    }
    if (status == cudaSuccess) status = cudaEventRecord(stop, nullptr);
    if (status == cudaSuccess) status = cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    if (status == cudaSuccess) status = cudaEventElapsedTime(&elapsed_ms, start, stop);
    if (stop != nullptr) cudaEventDestroy(stop);
    if (start != nullptr) cudaEventDestroy(start);

    double average_latency_us = 0.0;
    double effective_bandwidth_gbps = 0.0;
    if (status == cudaSuccess && elapsed_ms > 0.0f) {
        average_latency_us = static_cast<double>(elapsed_ms) * 1000.0 / iterations;
        effective_bandwidth_gbps =
            static_cast<double>(working_set_bytes) * iterations /
            (static_cast<double>(elapsed_ms) * 1.0e6);
    } else if (status == cudaSuccess) {
        status = cudaErrorUnknown;
    }

    std::string validation = "passed";
    if (status == cudaSuccess) {
        const int indices[] = {0, count / 2, count - 1};
        for (const int index : indices) {
            float actual = 0.0f;
            status = cudaMemcpy(&actual, device_c + index, sizeof(float), cudaMemcpyDeviceToHost);
            if (status != cudaSuccess) break;
            const float expected = host_a[static_cast<std::size_t>(index)] - host_b[static_cast<std::size_t>(index)];
            if (std::fabs(actual - expected) > 1e-6f) {
                validation = "failed_value_check";
                break;
            }
        }
    }
    if (status != cudaSuccess) {
        validation = std::string("failed:") + cudaGetErrorString(status);
    }

    cudaFree(device_c);
    cudaFree(device_b);
    cudaFree(device_a);
    std::size_t free_after_release = 0;
    std::size_t ignored_total_after_release = 0;
    (void)cudaMemGetInfo(&free_after_release, &ignored_total_after_release);
    print_result(device_name, runtime, driver, count, working_set_bytes,
                 free_before, total_bytes, free_after_alloc, free_after_release,
                 "allocated", average_latency_us, effective_bandwidth_gbps, validation);
    return validation == "passed" ? 0 : 1;
}
