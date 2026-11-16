#include "kernel_operator.h"
#include "acl/acl.h"

using namespace AscendC;

// Scaffold target: Ascend 310B (dav_m300). It has not been compiled or run because
// the board's installed CANN 7.0.RC1 package does not include dav_m300 headers.
extern "C" __global__ __aicore__ void color_inversion_kernel(GM_ADDR image, uint32_t pixel_count) {
    GlobalTensor<uint8_t> image_gm;
    image_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(image));

    const uint32_t block_index = static_cast<uint32_t>(GetBlockIdx());
    const uint32_t block_count = static_cast<uint32_t>(GetBlockNum());

    for (uint32_t pixel = block_index; pixel < pixel_count; pixel += block_count) {
        const uint32_t offset = pixel * 4;
        const uint8_t red = image_gm.GetValue(offset);
        const uint8_t green = image_gm.GetValue(offset + 1);
        const uint8_t blue = image_gm.GetValue(offset + 2);
        const uint8_t alpha = image_gm.GetValue(offset + 3);

        image_gm.SetBufferValue(offset, static_cast<uint8_t>(255U - red));
        image_gm.SetBufferValue(offset + 1, static_cast<uint8_t>(255U - green));
        image_gm.SetBufferValue(offset + 2, static_cast<uint8_t>(255U - blue));
        image_gm.SetBufferValue(offset + 3, alpha);
    }
}

void solve(uint8_t *image, int width, int height, aclrtStream stream) {
    const uint32_t pixel_count = static_cast<uint32_t>(width * height);
    constexpr uint32_t block_dim = 8;
    color_inversion_kernel<<<block_dim, nullptr, stream>>>(image, pixel_count);
}
