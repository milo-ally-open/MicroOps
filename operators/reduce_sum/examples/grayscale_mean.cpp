#include "ops/reduce_sum.hpp"

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

struct RgbPixel {
    float red;
    float green;
    float blue;
};

int main() {
    // A tiny in-memory RGB image keeps this demo independent of image libraries.
    const std::vector<RgbPixel> pixels = {
        {255.0F, 0.0F, 0.0F},
        {0.0F, 255.0F, 0.0F},
        {0.0F, 0.0F, 255.0F},
        {255.0F, 255.0F, 255.0F},
    };

    std::vector<float> grayscale;
    grayscale.reserve(pixels.size());
    for (const RgbPixel& pixel : pixels) {
        grayscale.push_back(0.299F * pixel.red +
                            0.587F * pixel.green +
                            0.114F * pixel.blue);
    }

    const float total_gray = ops::reduce_sum(grayscale.data(), grayscale.size());
    const float mean_gray = total_gray / static_cast<float>(grayscale.size());
    std::cout << std::fixed << std::setprecision(2)
              << "Pixels: " << pixels.size() << '\n'
              << "Gray sum (0-255 scale): " << total_gray << '\n'
              << "Gray mean (0-255 scale): " << mean_gray << '\n';
    return 0;
}
\n