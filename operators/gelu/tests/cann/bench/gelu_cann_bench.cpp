#include "ops/gelu_cann.hpp"
#include <acl/acl.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
float reference(float x) { return 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x))); }
int main(int argc,char**argv){
    const size_t n=argc>1?std::strtoull(argv[1],nullptr,10):1024;const int it=argc>2?std::atoi(argv[2]):100;if(!n||it<=0||n>2147483647u)return 2;
    if(aclInit(nullptr)!=ACL_SUCCESS||aclrtSetDevice(0)!=ACL_SUCCESS){std::puts("SKIP: CANN runtime/device unavailable.");return 77;}
    const char* device=aclrtGetSocName();if(!device)device="unknown";
    size_t f0=0,total=0,f1=0,f2=0;aclrtGetMemInfo(ACL_DDR_MEM,&f0,&total);const size_t bytes=n*sizeof(float),working=2*bytes;
    std::vector<float>x(n),y(n);for(size_t i=0;i<n;++i)x[i]=float(int(i%257)-128)/16.0f;
    float *dx=nullptr,*dy=nullptr;bool allocated=aclrtMalloc(reinterpret_cast<void**>(&dx),bytes,ACL_MEM_MALLOC_HUGE_FIRST)==ACL_SUCCESS&&aclrtMalloc(reinterpret_cast<void**>(&dy),bytes,ACL_MEM_MALLOC_HUGE_FIRST)==ACL_SUCCESS;aclrtGetMemInfo(ACL_DDR_MEM,&f1,&total);
    if(!allocated){if(dx)aclrtFree(dx);if(dy)aclrtFree(dy);aclrtGetMemInfo(ACL_DDR_MEM,&f2,&total);std::printf("CANN_RESULT\t%s\t%s\t%s\t%s\tnot-reported\t%zu\t%zu\t%zu\t%zu\t%zu\t%zu\tallocation_failed\t0\t0\tnot_run\n",GELU_CANN_KERNEL_FILE,GELU_CANN_HOST_FILE,device,GELU_CANN_RUNTIME_VERSION,n,working,f0,total,f1,f2);aclrtResetDevice(0);aclFinalize();return 0;}
    aclrtMemcpy(dx,bytes,x.data(),bytes,ACL_MEMCPY_HOST_TO_DEVICE);aclrtStream stream=nullptr;aclrtEvent start=nullptr,stop=nullptr;aclrtCreateStream(&stream);aclrtCreateEvent(&start);aclrtCreateEvent(&stop);
    for(int i=0;i<10;++i)ops::gelu_cann(dx,dy,int32_t(n),stream);aclrtSynchronizeStream(stream);aclrtRecordEvent(start,stream);for(int i=0;i<it;++i)ops::gelu_cann(dx,dy,int32_t(n),stream);aclrtRecordEvent(stop,stream);aclrtSynchronizeEvent(stop);float ms=0;aclrtEventElapsedTime(&ms,start,stop);
    const double us=double(ms)*1000/it,bw=double(working)/(us*1000);aclrtMemcpy(y.data(),bytes,dy,bytes,ACL_MEMCPY_DEVICE_TO_HOST);bool valid=true;for(size_t i=0;i<n;++i){const float e=reference(x[i]);if(!(std::fabs(y[i]-e)<=5e-5f + 5e-5f * fabsf(e))&&!(std::isnan(y[i])&&std::isnan(e))){valid=false;break;}}
    aclrtDestroyEvent(start);aclrtDestroyEvent(stop);aclrtDestroyStream(stream);aclrtFree(dx);aclrtFree(dy);aclrtGetMemInfo(ACL_DDR_MEM,&f2,&total);
    std::printf("CANN_RESULT\t%s\t%s\t%s\t%s\tnot-reported\t%zu\t%zu\t%zu\t%zu\t%zu\t%zu\tallocated\t%.6f\t%.6f\t%s\n",GELU_CANN_KERNEL_FILE,GELU_CANN_HOST_FILE,device,GELU_CANN_RUNTIME_VERSION,n,working,f0,total,f1,f2,us,bw,valid?"passed":"failed");aclrtResetDevice(0);aclFinalize();return valid?0:1;
}
