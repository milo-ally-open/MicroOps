# relu 规格

## 定义

对输入向量 `X` 的每个元素独立计算：`y = x if x > 0, otherwise y = 0; NaN is propagated.` 输出 `Y[i]`。

## 输入与输出

- `X`：连续 float32 设备输入；`Y`：至少容纳 N 个 float32 的设备输出；`N` 为元素数。
- 接口支持 `0 <= N <= INT_MAX`。`N == 0` 不访问指针且不启动 kernel；`N > 0` 时指针不可为空。
- host 对负长度或非零长度空指针返回后端非法参数错误；调用者负责分配、传输和同步。
- 不支持 stride、广播或破坏读取的输入输出重叠别名。

## dtype、shape 与版本

- 当前仅支持 float32，逻辑 shape 为 `[N]`。
- v1：一个线程处理一个元素。
- v2：每线程按四元素间隔展开并保护尾部；这是实现候选，不代表经测量的优化。
- CUDA/CANN 的当前设备验证状态见后文“后端状态”和 operator.yaml。

## 数值精度与边界

- 数学定义：y = x if x > 0, otherwise y = 0; NaN is propagated.
- 验收：exact for finite float32 inputs; NaN propagates。
- 测试覆盖 N=0、1、259、1027、负 N/空指针错误，以及正负输入、零点和饱和区样本；CUDA correctness 另用偏移设备指针检查对齐敏感路径。
- `relu` 对 NaN 传播；其他算子遵循指定公式及设备 float32 数学函数语义。输入测试以有限 float32 为主。

## 后端状态

- CUDA：v1/v2 已在 nvcc 12.6.85、CUDA Runtime 12.6 和 RTX 4060 Laptop GPU 上构建，并通过设备正确性、内存与 benchmark 检查。Nsight Compute 硬件计数器采集受 WSL 权限限制，不能据此声称 CUDA counter profile 完成。
- CANN：v1/v2 已在 Atlas 200I DK A2（310B4）、CANN 7.0.RC1 / dav-m300 上构建并通过设备正确性、内存与 benchmark 检查；可用板端 profile 由 msprof 采集。
- PyTorch 与 Triton：GPU-only 实现已在 RTX 4060 Laptop GPU 上通过 correctness、benchmark 和 torch.profiler profile。

## 验收标准

- 两组兼容 kernel/host 组合分别通过 correctness 测试，含尾部、未对齐输入和非法参数。
- 内存测试记录两个 float32 缓冲区的工作集、设备内存与分配状态。
- benchmark 记录设备、源码选择、输入规模、编译器/runtime、延迟、估算带宽和结果校验；未实测不填数值。

## PyTorch / Triton 实现状态

PyTorch 与 Triton 源码入口登记在 operator.yaml 的 frameworks.pytorch 和 frameworks.triton 下，均提供 CUDA GPU 上连续一维 float32 实现。两个后端已在 RTX 4060 Laptop GPU 上完成 test、bench 和 torch.profiler profile；性能记录位于各自 tests/pytorch/bench/results/ 与 tests/triton/bench/results/。
