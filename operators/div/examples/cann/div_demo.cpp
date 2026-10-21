#include "ops/div_cann.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    const int n = 1027;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    std::vector<float> a(n), b(n), c(n);
    for (int i = 0; i < n; ++i) { a[i] = float(i) / n; b[i] = float(i % 11) + 1.0f; }
    if (aclInit(nullptr) != ACL_SUCCESS || aclrtSetDevice(0) != ACL_SUCCESS) {
        std::puts("Could not initialize CANN device 0."); return 1;
    }
    aclrtStream stream = nullptr;
    float *da = nullptr, *db = nullptr, *dc = nullptr;
    bool ok = aclrtCreateStream(&stream) == ACL_SUCCESS &&
        aclrtMalloc(reinterpret_cast<void**>(&da), bytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS &&
        aclrtMalloc(reinterpret_cast<void**>(&db), bytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS &&
        aclrtMalloc(reinterpret_cast<void**>(&dc), bytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS;
    if (ok) {
        aclrtMemcpy(da, bytes, a.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(db, bytes, b.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        ok = ops::div_cann(da, db, dc, n, stream) == ACL_SUCCESS &&
             aclrtSynchronizeStream(stream) == ACL_SUCCESS &&
             aclrtMemcpy(c.data(), bytes, dc, bytes, ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
    }
    for (int i = 0; ok && i < n; ++i) {
        const float expected = a[i] / b[i];
        ok = std::fabs(c[i] - expected) <= 1e-5f * std::max(1.0f, std::fabs(expected));
    }
    std::printf("C = A / B vector demo: div_cann result=%s, elements=%d, sample output[0]=%.6f\n",
                ok ? "PASS" : "FAIL", n, ok ? c[0] : 0.0f);
    if (da) aclrtFree(da); if (db) aclrtFree(db); if (dc) aclrtFree(dc);
    if (stream) aclrtDestroyStream(stream);
    aclrtResetDevice(0); aclFinalize();
    return ok ? 0 : 1;
}
