#include <acl/acl.h>
#include <cstdio>
#include <cstdlib>
int main(int argc,char** argv) {
    const char* path=argc>1?argv[1]:"memory.csv"; const size_t n=argc>2?std::strtoull(argv[2],nullptr,10):65536;
    if(!n || n>static_cast<size_t>(-1)/(2*sizeof(float))) return 2;
    if(aclInit(nullptr)!=ACL_SUCCESS){std::puts("SKIP: CANN runtime unavailable.");return 77;}
    if(aclrtSetDevice(0)!=ACL_SUCCESS){std::puts("SKIP: Ascend device unavailable.");aclFinalize();return 77;}
    const char* device=aclrtGetSocName();if(!device)device="unknown";
    size_t before=0,total=0,after=0,released=0; aclrtGetMemInfo(ACL_DDR_MEM,&before,&total);
    const size_t one=n*sizeof(float); float *x=nullptr,*y=nullptr;
    const bool ok=aclrtMalloc(reinterpret_cast<void**>(&x),one,ACL_MEM_MALLOC_HUGE_FIRST)==ACL_SUCCESS &&
        aclrtMalloc(reinterpret_cast<void**>(&y),one,ACL_MEM_MALLOC_HUGE_FIRST)==ACL_SUCCESS;
    aclrtGetMemInfo(ACL_DDR_MEM,&after,&total); if(x)aclrtFree(x); if(y)aclrtFree(y);
    aclrtGetMemInfo(ACL_DDR_MEM,&released,&total);
    FILE* f=std::fopen(path,"a"); if(!f){aclrtResetDevice(0);aclFinalize();return 1;}
    if(std::ftell(f)==0)std::fprintf(f,"device,input_size,dtype,working_set_bytes,device_memory_total_bytes,device_memory_free_before_bytes,device_memory_free_after_alloc_bytes,device_memory_free_after_release_bytes,allocation_status\n");
    std::fprintf(f,"%s,%zu,float32,%zu,%zu,%zu,%zu,%zu,%s\n",device,n,2*one,total,before,after,released,ok?"allocated":"allocation_failed");
    std::fclose(f); aclrtResetDevice(0);aclFinalize(); return ok?0:77;
}
