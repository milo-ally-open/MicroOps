# add 规格

## 定义

给定长度为 N 的连续向量 A、B，输出满足 C[i] = A[i] + B[i]，其中 0 <= i < N。

## 输入与输出

- A、B：连续 float32 设备输入指针。
- C：至少容纳 N 个 float32 的设备输出指针。
- N：元素数；接口接受 0 <= N <= INT_MAX。
- N == 0 时不访问指针、不启动 kernel；N > 0 时必需指针不得为空。
- CUDA 与 CANN host wrapper 对负长度、非零长度空指针返回对应后端的非法参数错误。

## Dtype、shape 与别名

- 第一版仅支持 float32，逻辑 shape 为 [N]；多维数据由调用方展平。
- 不支持 stride 和广播；输出不得与输入发生破坏读取的重叠别名。
- CUDA v1 为标量一线程一元素；v2 对齐时使用 float4 并对未对齐输入回退到标量。
- CANN v1 使用 block_idx 和 block_num 的 grid-stride 循环逐元素计算。
- CANN v2 每轮展开四个独立的标量加法，尾部逐元素保护，以尝试增加指令级并行；不承诺编译成 SIMD 指令。
- 两个 CANN host 当前都在已验证的单 AI Core 板卡上发起单 block，均面向 dav-m300 / CANN 7.0 RC1。

## 数值精度

每个输出元素执行一次 IEEE 754 float32 加法。普通有限值以精确可表示用例或 abs(actual - expected) <= 1e-6 验收；NaN、Inf 和有符号零遵循设备浮点语义。

## 后端状态

- CUDA：v1/v2 已在 nvcc 12.6.85、CUDA Runtime 12.6 和 RTX 4060 Laptop GPU 上构建，并通过设备正确性、内存与 benchmark 检查。Nsight Compute 硬件计数器采集受 WSL 权限限制，不能据此声称 CUDA counter profile 完成。
- CANN：v1/v2 已在 Atlas 200I DK A2（310B4）、CANN 7.0.RC1 / dav-m300 上构建并通过设备正确性、内存与 benchmark 检查；可用板端 profile 由 msprof 采集。
- PyTorch 与 Triton：GPU-only 实现已在 RTX 4060 Laptop GPU 上通过 correctness、benchmark 和 torch.profiler profile。

## 验收标准

- 每组兼容的 kernel/host 文件都通过对应正确性工程；覆盖空输入、单元素、普通数据、尾部和非法参数。
- 内存评估记录输入规模、三缓冲理论申请量、设备总内存和分配前/后/释放后的可用量。
- 性能评估记录文件选择、设备、输入规模、dtype、编译器/runtime、迭代数、事件计时延迟、估算有效带宽和校验状态。有效带宽按 float32 加法每元素读两次、写一次估算 12*N 字节，不是硬件计数器测得的 DRAM 带宽。
- 缺少工具链时构建明确失败；缺少设备时设备测试显示 Skip，不计为通过。

## PyTorch / Triton 实现状态

PyTorch 与 Triton 源码入口登记在 operator.yaml 的 frameworks.pytorch 和 frameworks.triton 下，均提供 CUDA GPU 上连续一维 float32 实现。两个后端已在 RTX 4060 Laptop GPU 上完成 test、bench 和 torch.profiler profile；性能记录位于各自 tests/pytorch/bench/results/ 与 tests/triton/bench/results/。
