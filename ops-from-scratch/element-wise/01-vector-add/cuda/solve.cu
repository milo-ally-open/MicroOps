#include <cuda_runtime.h> 
#define CEIL(x, y) (((x) + (y) - 1) / (y))
const int THREADS = 256; 

/* Kernel 侧*/  
__device__ __forceinline__ float4 add4(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); 
}

__global__ void vector_add_kernel(const float* A, const float* B, float* C, int N) {
    
    // 线程模型 (获取全局线程 iD (tid) 和网格步长 (stride)
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x; 
    size_t stride = gridDim.x * blockDim.x; 

    // 计算 float4 的数据量
    size_t n4 = (size_t) N >> 2; 

    // 向量化
    const float4* A4 = reinterpret_cast<const float4*>(A); 
    const float4* B4 = reinterpret_cast<const float4*>(B); 
    float4* C4 = reinterpret_cast<float4*>(C); 

    // 主体计算
    for (size_t i = tid; i < n4; i += stride) {
        C4[i] = add4(A4[i], B4[i]); 
    }

    // 尾处理: 处理 N 不能被 4 整除时剩下的 1~3 个元素, 从 n4 * 4 的位置开始
    for (size_t i = n4 * 4 + tid; i < N; i += stride) {
        C[i] = A[i] + B[i]; 
    }
}

/* Host 侧*/
void solve(const float* A, const float* B, float* C, int N) {
    if (N <= 1) { 
        return; 
    }
    int n4 = (size_t) N >> 2;
    int block = CEIL(n4, THREADS); 
    vector_add_kernel<<<block, THREADS>>>(A, B, C, N);  
}