#include "acl/acl.h"
#include "kernel_operator.h"
#include <cstdint>

using namespace AscendC;

extern "C" __global__ __aicore__ void vector_add_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t count) {
    GlobalTensor<float> input_a, input_b, output;
    input_a.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a));
    input_b.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(b));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(c));
    const uint32_t first = static_cast<uint32_t>(GetBlockIdx());
    const uint32_t stride = static_cast<uint32_t>(GetBlockNum());
    for (uint32_t i = first; i < count; i += stride) {
        output.SetBufferValue(i, input_a.GetValue(i) + input_b.GetValue(i));
    }
}

extern "C" void vector_add_launch(void *a, void *b, void *c, uint32_t count, void *stream) {
    if (count == 0) return;
    constexpr uint32_t block_dim = 8;
    vector_add_kernel<<<block_dim, nullptr, reinterpret_cast<aclrtStream>(stream)>>>(
        reinterpret_cast<GM_ADDR>(a), reinterpret_cast<GM_ADDR>(b),
        reinterpret_cast<GM_ADDR>(c), count);
}
