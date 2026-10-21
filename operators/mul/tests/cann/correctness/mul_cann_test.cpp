#include "ops/mul_cann.hpp"
#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    const aclError init = aclInit(nullptr);
    if (init != ACL_SUCCESS) { std::puts("SKIP: aclInit failed; CANN runtime unavailable."); return 77; }
    if (aclrtSetDevice(0) != ACL_SUCCESS) { std::puts("SKIP: no usable Ascend device 0."); aclFinalize(); return 77; }
    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) { std::puts("SKIP: could not create ACL stream."); aclrtResetDevice(0); aclFinalize(); return 77; }

    bool ok = true;
    for (int n : {0, 1, 259, 1027}) {
        if (n == 0) {
            ok = ok && ops::mul_cann(nullptr, nullptr, nullptr, 0, stream) == ACL_SUCCESS;
            continue;
        }
        const size_t bytes = static_cast<size_t>(n) * sizeof(float);
        std::vector<float> a(n), b(n), c(n, 0.0f);
        for (int i = 0; i < n; ++i) { a[i] = float(i) * 0.25f; b[i] = float(i % 17) - 3.0f; }
        float *da = nullptr, *db = nullptr, *dc = nullptr;
        if (aclrtMalloc(reinterpret_cast<void**>(&da), bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
            aclrtMalloc(reinterpret_cast<void**>(&db), bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
            aclrtMalloc(reinterpret_cast<void**>(&dc), bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            std::printf("SKIP: device allocation failed for N=%d.\n", n); ok = false;
        } else {
            aclrtMemcpy(da, bytes, a.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
            aclrtMemcpy(db, bytes, b.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
            ok = ok && ops::mul_cann(da, db, dc, n, stream) == ACL_SUCCESS;
            ok = ok && aclrtSynchronizeStream(stream) == ACL_SUCCESS;
            ok = ok && aclrtMemcpy(c.data(), bytes, dc, bytes, ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
            for (int i = 0; i < n; ++i) if (c[i] != a[i] * b[i]) { std::printf("mismatch N=%d index=%d\n", n, i); ok = false; break; }
        }
        if (da) aclrtFree(da); if (db) aclrtFree(db); if (dc) aclrtFree(dc);
    }
    ok = ok && ops::mul_cann(nullptr, nullptr, nullptr, -1, stream) == ACL_ERROR_INVALID_PARAM;
    ok = ok && ops::mul_cann(nullptr, nullptr, nullptr, 1, stream) == ACL_ERROR_INVALID_PARAM;
    aclrtDestroyStream(stream); aclrtResetDevice(0); aclFinalize();
    std::puts(ok ? "CANN correctness PASS" : "CANN correctness FAIL");
    return ok ? 0 : 1;
}
