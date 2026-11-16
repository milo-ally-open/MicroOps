#include <cuda_runtime.h>

#define CEIL(x, y) (((x) + (y) - 1) / (y))

const int THREADS = 256; 

__device__ __forceinline__ uchar4 invertColor(uchar4 v) {

}

__global__ void color_version_kernel(unsigned char* image, int nPix) {
    
    // 线程模型 


    // 向量化 


    // 主体计算 

    // 尾处理 None
}

void solve(unsigned char* image, int width, int height) {

}
