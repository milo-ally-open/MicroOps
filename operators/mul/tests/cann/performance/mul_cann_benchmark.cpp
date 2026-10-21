#include "ops/mul_cann.hpp"
#ifndef MUL_CANN_KERNEL_FILE
#define MUL_CANN_KERNEL_FILE "unknown"
#define MUL_CANN_HOST_FILE "unknown"
#define MUL_CANN_RUNTIME_VERSION "unknown"
#endif
#include <acl/acl.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    const size_t n = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1024;
    const int iterations = argc > 2 ? std::atoi(argv[2]) : 100;
    if (!n || iterations <= 0 || n > 2147483647u) return 2;
    if (aclInit(nullptr) != ACL_SUCCESS || aclrtSetDevice(0) != ACL_SUCCESS) {
        std::puts("SKIP: CANN runtime or device 0 unavailable."); return 77;
    }
    size_t free_before = 0, total = 0, free_alloc = 0, free_release = 0;
    aclrtGetMemInfo(ACL_DDR_MEM, &free_before, &total);
    const size_t bytes = n * sizeof(float);
    std::vector<float> a(n, 1.25f), b(n, 2.5f), c(n, 0.0f);
    float *da = nullptr, *db = nullptr, *dc = nullptr;
    if (aclrtMalloc(reinterpret_cast<void**>(&da), bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMalloc(reinterpret_cast<void**>(&db), bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMalloc(reinterpret_cast<void**>(&dc), bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        if(da)aclrtFree(da); if(db)aclrtFree(db); if(dc)aclrtFree(dc); aclrtResetDevice(0); aclFinalize();
        std::puts("SKIP: device allocation failed."); return 77;
    }
    aclrtGetMemInfo(ACL_DDR_MEM, &free_alloc, &total);
    aclrtStream stream = nullptr;
    aclrtEvent start = nullptr, stop = nullptr;
    aclrtCreateStream(&stream); aclrtCreateEvent(&start); aclrtCreateEvent(&stop);
    aclrtMemcpy(da, bytes, a.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(db, bytes, b.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    ops::mul_cann(da, db, dc, static_cast<int32_t>(n), stream);
    aclrtSynchronizeStream(stream);
    aclrtRecordEvent(start, stream);
    for (int i = 0; i < iterations; ++i) ops::mul_cann(da, db, dc, static_cast<int32_t>(n), stream);
    aclrtRecordEvent(stop, stream);
    aclrtSynchronizeEvent(stop);
    float elapsed_ms = 0.0f;
    aclrtEventElapsedTime(&elapsed_ms, start, stop);
    const double avg_us = double(elapsed_ms) * 1000.0 / iterations;
    aclrtMemcpy(c.data(), bytes, dc, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    bool valid = true;
    for (size_t i = 0; i < n; ++i) if (c[i] != a[i] * b[i]) { valid = false; break; }
    aclrtDestroyEvent(start); aclrtDestroyEvent(stop); aclrtDestroyStream(stream);
    aclrtFree(da); aclrtFree(db); aclrtFree(dc);
    aclrtGetMemInfo(ACL_DDR_MEM, &free_release, &total);
    const double bandwidth = (double(12 * n) / (avg_us * 1.0e3));
    const char* soc_name = aclrtGetSocName();
    std::printf("CANN_RESULT\t%s\t%s\t%s\t%s\t%s\t%zu\t%zu\t%zu\t%zu\t%zu\t%zu\tallocated\t%.6f\t%.6f\t%s\n",
        MUL_CANN_KERNEL_FILE, MUL_CANN_HOST_FILE, soc_name ? soc_name : "unknown-soc", MUL_CANN_RUNTIME_VERSION, "not-reported", n, bytes*3, free_before, total, free_alloc, free_release, avg_us, bandwidth, valid ? "passed" : "failed");
    aclrtResetDevice(0); aclFinalize();
    return valid ? 0 : 1;
}
