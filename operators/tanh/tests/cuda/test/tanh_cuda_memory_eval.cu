#include <cuda_runtime.h>
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
int main(int argc, char** argv) {
    if (argc != 3) { std::cerr << "Usage: tanh_cuda_memory_eval <results.csv> <positive_N>\n"; return 2; }
    const size_t n = static_cast<size_t>(std::strtoull(argv[2], nullptr, 10));
    if (!n || n > static_cast<size_t>(-1) / (2 * sizeof(float))) return 2;
    int count = 0; cudaError_t s = cudaGetDeviceCount(&count);
    if (s != cudaSuccess || count == 0) { std::cout << "SKIP: CUDA device/driver unavailable\n"; return 77; }
    cudaDeviceProp p{}; int id = 0, rv = 0, dv = 0;
    if ((s=cudaGetDevice(&id))!=cudaSuccess || (s=cudaGetDeviceProperties(&p,id))!=cudaSuccess ||
        (s=cudaRuntimeGetVersion(&rv))!=cudaSuccess || (s=cudaDriverGetVersion(&dv))!=cudaSuccess) return 1;
    size_t free0=0,total=0,free1=0; cudaMemGetInfo(&free0,&total);
    const size_t bytes=n*sizeof(float), working=2*bytes; float *x=nullptr,*y=nullptr;
    s=cudaMalloc(reinterpret_cast<void**>(&x),bytes); if(s==cudaSuccess)s=cudaMalloc(reinterpret_cast<void**>(&y),bytes);
    cudaMemGetInfo(&free1,&total); if(x)cudaFree(x); if(y)cudaFree(y); size_t free2=0; cudaMemGetInfo(&free2,&total);
    std::filesystem::path path(argv[1]); std::filesystem::create_directories(path.parent_path());
    std::ofstream f(path,std::ios::app); if(!f)return 1;
    if(f.tellp()==0) f<<"backend,device,input_elements,dtype,compiler,compiler_version,runtime_version,driver_version,requested_bytes,device_memory_total_bytes,device_memory_free_before_bytes,device_memory_free_after_alloc_bytes,device_memory_free_after_release_bytes,allocation_status\n";
    f<<"cuda,\""<<p.name<<"\","<<n<<",float32,nvcc,"<<TANH_CUDA_COMPILER_VERSION<<","<<rv<<","<<dv<<","<<working<<","<<total<<","<<free0<<","<<free1<<","<<free2<<","<<(s==cudaSuccess?"allocated":cudaGetErrorString(s))<<"\n";
    std::cout<<"CUDA memory evaluation: N="<<n<<" requested_bytes="<<working<<"\n"; return s==cudaSuccess?0:1;
}
