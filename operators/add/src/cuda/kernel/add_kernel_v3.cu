// Size-aware dispatch implementation: include the simple one-element path
// and the aligned float4 path so the host can select between them by N.
#include "add_kernel_v1.cu"
#include "add_kernel_v2.cu"
