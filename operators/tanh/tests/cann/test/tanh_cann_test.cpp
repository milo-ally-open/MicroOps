#include "ops/tanh_cann.hpp"
#include <acl/acl.h>
#include <cmath>
#include <cstdio>
#include <vector>
float reference(float x) { return std::tanh(x); }
int main() {
    if (aclInit(nullptr) != ACL_SUCCESS) { std::puts("SKIP: CANN runtime unavailable."); return 77; }
    if (aclrtSetDevice(0) != ACL_SUCCESS) { std::puts("SKIP: no usable Ascend device 0."); aclFinalize(); return 77; }
    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) { std::puts("SKIP: could not create ACL stream."); aclrtResetDevice(0); aclFinalize(); return 77; }
    bool ok = true;
    const float samples[] = {-8.0f, -3.0f, -0.5f, 0.0f, 0.5f, 3.0f, 8.0f};
    for (int n : {0, 1, 259, 1027}) {
        if (n == 0) { ok = ok && ops::tanh_cann(nullptr, nullptr, 0, stream) == ACL_SUCCESS; continue; }
        const size_t bytes = static_cast<size_t>(n) * sizeof(float);
        std::vector<float> x(n), y(n, 0.0f);
        for (int i = 0; i < n; ++i) x[i] = samples[i % (sizeof(samples) / sizeof(samples[0]))];
        float *dx = nullptr, *dy = nullptr;
        if (aclrtMalloc(reinterpret_cast<void**>(&dx), bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
            aclrtMalloc(reinterpret_cast<void**>(&dy), bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            std::puts("SKIP: device allocation failed."); ok = false;
        } else {
            ok = ok && aclrtMemcpy(dx, bytes, x.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
            ok = ok && ops::tanh_cann(dx, dy, n, stream) == ACL_SUCCESS;
            ok = ok && aclrtSynchronizeStream(stream) == ACL_SUCCESS;
            ok = ok && aclrtMemcpy(y.data(), bytes, dy, bytes, ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
            for (int i = 0; ok && i < n; ++i) { const float expected = reference(x[i]);
                if (!(std::fabs(y[i] - expected) <= 2e-6f) && !(std::isnan(y[i]) && std::isnan(expected))) {
                    std::printf("mismatch N=%d i=%d expected=%g got=%g\n", n, i, expected, y[i]); ok = false;
                }
            }
        }
        if (dx) aclrtFree(dx); if (dy) aclrtFree(dy);
    }
    ok = ok && ops::tanh_cann(nullptr, nullptr, -1, stream) == ACL_ERROR_INVALID_PARAM;
    ok = ok && ops::tanh_cann(nullptr, nullptr, 1, stream) == ACL_ERROR_INVALID_PARAM;
    aclrtDestroyStream(stream); aclrtResetDevice(0); aclFinalize();
    std::puts(ok ? "CANN correctness PASS" : "CANN correctness FAIL"); return ok ? 0 : 1;
}
