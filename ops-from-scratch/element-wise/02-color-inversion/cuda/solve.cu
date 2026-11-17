#include <cuda_runtime.h>

#define CEIL(x, y) (((x) + (y) - 1) / (y))

const int THREADS = 256; 

__device__ __forceinline__ uchar4 invertColor(uchar4 c) {
    return make_uchar4(255 - c.x, 255 - c.y, 255 - c.z, c.w); 
} 

__global__ void color_inversion_kernel(unsigned char* image, int pixels) {
    // CUDA 线程模型 
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x; 
    size_t stride = blockDim.x * gridDim.x; 
    // 向量化 
    uchar4* c4 = reinterpret_cast<uchar4*>(image);  
    // 主计算
    for (size_t i = tid; i < (size_t) pixels; i += stride) {
        c4[i] = invertColor(c4[i]);
    }
    // 尾处理: None 
}

void solve(unsigned char* image, int width, int height) {
    int pixels = width * height; 
    int block = CEIL(pixels, THREADS); 
    color_inversion_kernel<<<block, THREADS>>>(image, pixels);
}