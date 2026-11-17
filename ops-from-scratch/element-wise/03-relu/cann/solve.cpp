#include "acl/acl.h"
#include "kernel_operator.h"
#include <cstdint>

using namespace AscendC;

extern "C" __global__ __aicore__ void relu_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t count) {
    GlobalTensor<float> input_tensor, output_tensor;
    input_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(input));
    output_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output));

    const uint32_t first = static_cast<uint32_t>(GetBlockIdx());
    const uint32_t stride = static_cast<uint32_t>(GetBlockNum());
    for (uint32_t i = first; i < count; i += stride) {
        const float value = input_tensor.GetValue(i);
        output_tensor.SetBufferValue(i, value > 0.0f ? value : 0.0f);
    }
}

extern "C" void relu_launch(
    void *input, void *output, uint32_t count, void *stream) {
    if (count == 0) return;
    constexpr uint32_t block_dim = 8;
    relu_kernel<<<block_dim, nullptr, reinterpret_cast<aclrtStream>(stream)>>>(
        reinterpret_cast<GM_ADDR>(input), reinterpret_cast<GM_ADDR>(output), count);
}
