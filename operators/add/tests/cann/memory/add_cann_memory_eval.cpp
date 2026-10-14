#include <acl/acl.h>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "memory.csv";
    const size_t n = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 65536;
    if (aclInit(nullptr) != ACL_SUCCESS || aclrtSetDevice(0) != ACL_SUCCESS) {
        std::puts("SKIP: CANN runtime or device 0 unavailable."); return 77;
    }
    size_t free_before = 0, total = 0, free_alloc = 0, free_release = 0;
    aclrtGetMemInfo(ACL_DDR_MEM, &free_before, &total);
    const size_t one = n * sizeof(float);
    float *a = nullptr, *b = nullptr, *c = nullptr;
    const bool allocated =
        aclrtMalloc(reinterpret_cast<void**>(&a), one, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS &&
        aclrtMalloc(reinterpret_cast<void**>(&b), one, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS &&
        aclrtMalloc(reinterpret_cast<void**>(&c), one, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS;
    if (!allocated) { if(a)aclrtFree(a); if(b)aclrtFree(b); if(c)aclrtFree(c); aclrtResetDevice(0); aclFinalize(); std::puts("SKIP: device memory allocation failed."); return 77; }
    aclrtGetMemInfo(ACL_DDR_MEM, &free_alloc, &total);
    aclrtFree(a); aclrtFree(b); aclrtFree(c);
    aclrtGetMemInfo(ACL_DDR_MEM, &free_release, &total);
    std::FILE* f = std::fopen(path, "a");
    if (!f) { std::perror("open memory CSV"); aclrtResetDevice(0); aclFinalize(); return 1; }
    if (std::ftell(f) == 0) std::fprintf(f, "device,input_size,dtype,working_set_bytes,device_memory_total_bytes,device_memory_free_before_bytes,device_memory_free_after_alloc_bytes,device_memory_free_after_release_bytes,allocation_status\n");
    std::fprintf(f, "Ascend-310B4,%zu,float32,%zu,%zu,%zu,%zu,%zu,allocated\n", n, one * 3, total, free_before, free_alloc, free_release);
    std::fclose(f);
    std::printf("CANN memory record: N=%zu working_set=%zu total=%zu free_before=%zu free_after_alloc=%zu free_after_release=%zu\n", n, one*3, total, free_before, free_alloc, free_release);
    aclrtResetDevice(0); aclFinalize(); return 0;
}
