#include "ops/tanh_cuda.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
float reference(float x) { return std::tanh(x); }
int main(int argc,char** argv) {
    if(argc!=3){std::cerr<<"Usage: tanh_cuda_benchmark <N> <iterations>\n";return 2;}
    const int n=std::atoi(argv[1]), iterations=std::atoi(argv[2]); if(n<=0||iterations<=0)return 2;
    int id=0,rv=0,dv=0; cudaDeviceProp prop{}; cudaError_t s=cudaGetDevice(&id);
    if(s==cudaSuccess)s=cudaGetDeviceProperties(&prop,id); if(s==cudaSuccess)s=cudaRuntimeGetVersion(&rv); if(s==cudaSuccess)s=cudaDriverGetVersion(&dv);
    if(s!=cudaSuccess){std::cerr<<cudaGetErrorString(s)<<'\n';return 1;}
    const size_t bytes=size_t(n)*sizeof(float),working=2*bytes; size_t free0=0,total=0,free1=0,free2=0; cudaMemGetInfo(&free0,&total);
    float *dx=nullptr,*dy=nullptr; s=cudaMalloc(reinterpret_cast<void**>(&dx),bytes);if(s==cudaSuccess)s=cudaMalloc(reinterpret_cast<void**>(&dy),bytes);cudaMemGetInfo(&free1,&total);
    if(s!=cudaSuccess){if(dx)cudaFree(dx);if(dy)cudaFree(dy);cudaMemGetInfo(&free2,&total);std::cout<<std::setprecision(9)<<"CUDA_RESULT\t"<<TANH_CUDA_KERNEL_FILE<<'\t'<<TANH_CUDA_HOST_FILE<<'\t'<<prop.name<<"\t"<<rv<<"\t"<<dv<<'\t'<<n<<'\t'<<working<<'\t'<<free0<<'\t'<<total<<'\t'<<free1<<'\t'<<free2<<"\tallocation_failed\t0\t0\tnot_run\n";return 0;}
    std::vector<float>x(n),y(n);for(int i=0;i<n;++i)x[i]=float((i%257)-128)/16.0f;
    s=cudaMemcpy(dx,x.data(),bytes,cudaMemcpyHostToDevice);for(int i=0;s==cudaSuccess&&i<10;++i)s=ops::tanh_cuda(dx,dy,n,nullptr);if(s==cudaSuccess)s=cudaDeviceSynchronize();
    cudaEvent_t start=nullptr,stop=nullptr;if(s==cudaSuccess)s=cudaEventCreate(&start);if(s==cudaSuccess)s=cudaEventCreate(&stop);if(s==cudaSuccess)s=cudaEventRecord(start);
    for(int i=0;s==cudaSuccess&&i<iterations;++i)s=ops::tanh_cuda(dx,dy,n,nullptr);if(s==cudaSuccess)s=cudaEventRecord(stop);if(s==cudaSuccess)s=cudaEventSynchronize(stop);
    float ms=0;if(s==cudaSuccess)s=cudaEventElapsedTime(&ms,start,stop);if(start)cudaEventDestroy(start);if(stop)cudaEventDestroy(stop);
    double us=s==cudaSuccess?double(ms)*1000/iterations:0,bw=us>0?double(working)/(us*1000):0;
    if(s==cudaSuccess)s=cudaMemcpy(y.data(),dy,bytes,cudaMemcpyDeviceToHost);bool valid=s==cudaSuccess;
    for(int i=0;valid&&i<n;++i){float e=reference(x[i]);if(!(std::fabs(y[i]-e)<=2e-6f)&&!(std::isnan(y[i])&&std::isnan(e)))valid=false;}
    cudaFree(dx);cudaFree(dy);cudaMemGetInfo(&free2,&total);
    std::cout<<std::setprecision(9)<<"CUDA_RESULT\t"<<TANH_CUDA_KERNEL_FILE<<'\t'<<TANH_CUDA_HOST_FILE<<'\t'<<prop.name<<"\t"<<rv<<"\t"<<dv<<'\t'<<n<<'\t'<<working<<'\t'<<free0<<'\t'<<total<<'\t'<<free1<<'\t'<<free2<<"\tallocated\t"<<us<<'\t'<<bw<<'\t'<<(valid?"passed":"failed")<<'\n';return valid?0:1;
}
