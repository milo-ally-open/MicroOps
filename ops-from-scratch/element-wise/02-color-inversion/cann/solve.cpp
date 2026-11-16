#include "acl/acl.h"
#include "kernel_operator.h"
#include <cstdint>

using namespace AscendC;

extern "C" __global__ __aicore__ void color_inversion_kernel(
    GM_ADDR image, uint32_t pixel_count) {
    GlobalTensor<uint8_t> pixels;
    pixels.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(image));
    const uint32_t first = static_cast<uint32_t>(GetBlockIdx());
    const uint32_t stride = static_cast<uint32_t>(GetBlockNum());
    for (uint32_t pixel = first; pixel < pixel_count; pixel += stride) {
        const uint32_t offset = pixel * 4;
        const uint8_t red = pixels.GetValue(offset);
        const uint8_t green = pixels.GetValue(offset + 1);
        const uint8_t blue = pixels.GetValue(offset + 2);
        const uint8_t alpha = pixels.GetValue(offset + 3);
        pixels.SetBufferValue(offset, static_cast<uint8_t>(255U - red));
        pixels.SetBufferValue(offset + 1, static_cast<uint8_t>(255U - green));
        pixels.SetBufferValue(offset + 2, static_cast<uint8_t>(255U - blue));
        pixels.SetBufferValue(offset + 3, alpha);
    }
}

extern "C" void color_inversion_launch(void *image, uint32_t width, uint32_t height, void *stream) {
    if (width == 0 || height == 0) return;
    constexpr uint32_t block_dim = 8;
    color_inversion_kernel<<<block_dim, nullptr, reinterpret_cast<aclrtStream>(stream)>>>(
        reinterpret_cast<GM_ADDR>(image), width * height);
}
