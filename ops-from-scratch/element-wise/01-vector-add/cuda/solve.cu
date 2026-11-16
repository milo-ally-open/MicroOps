#include <cuda_runtime.h> 
#define CEIL(x, y) (((x) + (y) - 1) / (y))
const int THREADS = 256; 

/* Kernel 侧*/  
__device__ __forceinline__ float4 add4(float4 a, float4 b) {

}

__global__ void vector_add_kernel(const float* A, const float* B, float* C, int N) {
    
    // 线程模型 (获取全局线程 iD (tid) 和网格步长 (stride)


    // 计算 float4 的数据量


    // 向量化


    // 主体计算


    // 尾处理: 处理 N 不能被 4 整除时剩下的 1~3 个元素, 从 n4 * 4 的位置开始

}

/* Host 侧*/
void solve(const float* A, const float* B, float* C, int N) {

}