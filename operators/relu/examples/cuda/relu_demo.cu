#include "ops/relu_cuda.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <vector>
int main() {
    std::vector<float> x{-2.0f, -0.5f, 0.0f, 0.5f, 2.0f}, y(x.size());
    const size_t bytes=x.size()*sizeof(float);float *dx=nullptr,*dy=nullptr;
    cudaError_t s=cudaMalloc(reinterpret_cast<void**>(&dx),bytes);if(s==cudaSuccess)s=cudaMalloc(reinterpret_cast<void**>(&dy),bytes);
    if(s==cudaSuccess)s=cudaMemcpy(dx,x.data(),bytes,cudaMemcpyHostToDevice);
    if(s==cudaSuccess)s=ops::relu_cuda(dx,dy,int(x.size()),nullptr);if(s==cudaSuccess)s=cudaDeviceSynchronize();
    if(s==cudaSuccess)s=cudaMemcpy(y.data(),dy,bytes,cudaMemcpyDeviceToHost);
    if(dy)cudaFree(dy);if(dx)cudaFree(dx);if(s!=cudaSuccess){std::cerr<<cudaGetErrorString(s)<<'\n';return 1;}
    for(size_t i=0;i<x.size();++i) { const float expected=x[i]>0.0f?x[i]:0.0f; if(y[i]!=expected)return 1; }
    std::cout<<"CUDA relu: ";for(float value:y)std::cout<<value<<' ';std::cout<<'\n';return 0;
}
