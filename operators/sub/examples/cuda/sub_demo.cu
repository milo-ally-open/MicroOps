#include "ops/sub_cuda.hpp"

#include <cuda_runtime.h>

#include <iostream>
#include <vector>

int main() {
    const std::vector<float> host_a{1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> host_b{0.5f, -1.0f, 2.0f, 8.0f};
    std::vector<float> host_c(host_a.size());
    const std::size_t bytes = host_a.size() * sizeof(float);
    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_c = nullptr;

    cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&device_a), bytes);
    if (status == cudaSuccess) status = cudaMalloc(reinterpret_cast<void**>(&device_b), bytes);
    if (status == cudaSuccess) status = cudaMalloc(reinterpret_cast<void**>(&device_c), bytes);
    if (status == cudaSuccess) status = cudaMemcpy(device_a, host_a.data(), bytes, cudaMemcpyHostToDevice);
    if (status == cudaSuccess) status = cudaMemcpy(device_b, host_b.data(), bytes, cudaMemcpyHostToDevice);
    if (status == cudaSuccess) {
        status = ops::sub_cuda(device_a, device_b, device_c,
                               static_cast<int>(host_a.size()), nullptr);
    }
    if (status == cudaSuccess) status = cudaDeviceSynchronize();
    if (status == cudaSuccess) status = cudaMemcpy(host_c.data(), device_c, bytes, cudaMemcpyDeviceToHost);

    if (device_c != nullptr) cudaFree(device_c);
    if (device_b != nullptr) cudaFree(device_b);
    if (device_a != nullptr) cudaFree(device_a);
    if (status != cudaSuccess) {
        std::cerr << "CUDA sub demo failed: " << cudaGetErrorString(status) << '\n';
        return 1;
    }

    std::cout << "CUDA C = A - B: ";
    for (const float value : host_c) std::cout << value << ' ';
    std::cout << '\n';
    return 0;
}
