#include "acl/acl.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

void solve(uint8_t *image, int width, int height, aclrtStream stream);

void check_acl(aclError status, const char *operation) {
    if (status != ACL_SUCCESS) {
        std::cerr << operation << " failed with ACL error " << status << '\n';
        std::exit(EXIT_FAILURE);
    }
}

int main() {
    constexpr int width = 1025;
    constexpr int height = 2049;
    constexpr int iterations = 100;
    constexpr int device_id = 0;
    const size_t pixel_count = static_cast<size_t>(width) * height;
    const size_t image_bytes = pixel_count * 4;

    check_acl(aclInit(nullptr), "aclInit");
    check_acl(aclrtSetDevice(device_id), "aclrtSetDevice");

    aclrtContext context = nullptr;
    check_acl(aclrtCreateContext(&context, device_id), "aclrtCreateContext");
    aclrtStream stream = nullptr;
    check_acl(aclrtCreateStream(&stream), "aclrtCreateStream");

    std::vector<uint8_t> image(image_bytes);
    std::vector<uint8_t> expected(image_bytes);
    for (size_t i = 0; i < image_bytes; ++i) {
        image[i] = static_cast<uint8_t>((i * 37 + 11) % 256);
        expected[i] = (i % 4 == 3) ? image[i] : static_cast<uint8_t>(255 - image[i]);
    }

    uint8_t *image_device = nullptr;
    check_acl(aclrtMalloc(reinterpret_cast<void **>(&image_device), image_bytes,
                          ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
    check_acl(aclrtMemcpy(image_device, image_bytes, image.data(), image_bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE), "copy image to device");

    solve(image_device, width, height, stream);
    check_acl(aclrtSynchronizeStream(stream), "sync after correctness run");

    std::vector<uint8_t> actual(image_bytes);
    check_acl(aclrtMemcpy(actual.data(), image_bytes, image_device, image_bytes,
                          ACL_MEMCPY_DEVICE_TO_HOST), "copy result to host");
    for (size_t i = 0; i < image_bytes; ++i) {
        if (actual[i] != expected[i]) {
            std::cerr << "test failed at byte " << i << ": expected "
                      << static_cast<int>(expected[i]) << ", got "
                      << static_cast<int>(actual[i]) << '\n';
            return EXIT_FAILURE;
        }
    }
    std::cout << "test successfully! shape=(" << height << ", " << width
              << ", 4), RGB inverted, alpha preserved\n";

    for (int i = 0; i < 10; ++i) {
        solve(image_device, width, height, stream);
    }
    check_acl(aclrtSynchronizeStream(stream), "sync after warmup");

    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        solve(image_device, width, height, stream);
    }
    check_acl(aclrtSynchronizeStream(stream), "sync after benchmark");
    const auto stop = std::chrono::steady_clock::now();

    const double elapsed_seconds = std::chrono::duration<double>(stop - start).count();
    const double latency_us = elapsed_seconds * 1e6 / iterations;
    const double estimated_gbps = 8.0 * pixel_count * iterations / elapsed_seconds / 1e9;
    std::cout << "device id: " << device_id << ", input: " << width << "x" << height
              << " RGBA, iterations: " << iterations << '\n';
    std::cout << "custom CANN kernel: " << latency_us << " us, estimated read+write bandwidth: "
              << estimated_gbps << " GB/s\n";

    check_acl(aclrtFree(image_device), "aclrtFree");
    check_acl(aclrtDestroyStream(stream), "aclrtDestroyStream");
    check_acl(aclrtDestroyContext(context), "aclrtDestroyContext");
    check_acl(aclrtResetDevice(device_id), "aclrtResetDevice");
    check_acl(aclFinalize(), "aclFinalize");
    return EXIT_SUCCESS;
}
