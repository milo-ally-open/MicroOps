#include "ops/silu_cann.hpp"
#include <cmath>
#include <cstdio>
#include <vector>
int main() {
    std::vector<float>x{-2.0f,-0.5f,0.0f,0.5f,2.0f},y(x.size());const size_t bytes=x.size()*sizeof(float);
    if(aclInit(nullptr)!=ACL_SUCCESS||aclrtSetDevice(0)!=ACL_SUCCESS){std::puts("Could not initialize CANN device 0.");return 1;}
    aclrtStream stream=nullptr;float *dx=nullptr,*dy=nullptr;
    bool ok=aclrtCreateStream(&stream)==ACL_SUCCESS&&aclrtMalloc(reinterpret_cast<void**>(&dx),bytes,ACL_MEM_MALLOC_HUGE_FIRST)==ACL_SUCCESS&&aclrtMalloc(reinterpret_cast<void**>(&dy),bytes,ACL_MEM_MALLOC_HUGE_FIRST)==ACL_SUCCESS;
    if(ok){ok=aclrtMemcpy(dx,bytes,x.data(),bytes,ACL_MEMCPY_HOST_TO_DEVICE)==ACL_SUCCESS&&ops::silu_cann(dx,dy,int(x.size()),stream)==ACL_SUCCESS&&aclrtSynchronizeStream(stream)==ACL_SUCCESS&&aclrtMemcpy(y.data(),bytes,dy,bytes,ACL_MEMCPY_DEVICE_TO_HOST)==ACL_SUCCESS;}
    for(size_t i=0;ok&&i<x.size();++i){const float e=x[i]/(1.0f+std::exp(-x[i]));ok=std::fabs(y[i]-e)<=2e-5f+2e-5f*std::fabs(e);}
    std::printf("CANN silu %s: ",ok?"result":"failed");for(float v:y)std::printf("%.6f ",v);std::puts("");
    if(dx)aclrtFree(dx);if(dy)aclrtFree(dy);if(stream)aclrtDestroyStream(stream);aclrtResetDevice(0);aclFinalize();return ok?0:1;
}
