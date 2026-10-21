# mul 规格

## 定义

给定长度为 `N` 的连续向量 `A`、`B`，输出 `C[i] = A[i] * B[i]`，其中 `0 <= i < N`。算子名 `mul` 表示逐元素乘法，不是归约。

## 输入与输出

- `A`、`B`：连续 float32 设备输入指针。
- `C`：至少容纳 `N` 个 float32 的设备输出指针。
- `N`：元素数；接口接受 `0 <= N <= INT_MAX`。
- `N == 0` 时不访问指针、不启动 kernel；`N > 0` 时必需指针不得为空。
- CUDA/CANN host 对负长度和非零长度空指针返回对应后端非法参数错误。

## Dtype、shape 与实现版本

- 仅支持 float32，逻辑 shape 为 `[N]`；多维数据由调用方展平。
- 不支持 stride、广播或会破坏输入读取的重叠别名。
- CUDA v1 为标量一线程一元素；v2 对齐时使用 float4、未对齐时回退标量并正确处理尾部。
- CANN v1 使用 `block_idx`/`block_num` grid-stride 循环；v2 每轮展开四个标量元素并保护尾部，不声称使用 SIMD 指令。

## 数值精度

有限值按 `abs(actual - expected) <= 1e-6 * max(1, abs(expected))` 比较；NaN、Inf 和有符号零遵循目标设备浮点语义。

## 后端状态

- CUDA：kernel、host、build、correctness、显存评估、benchmark 和 demo 已建立为 scaffold；本仓库尚未确认此算子通过 nvcc 构建或任何 CUDA 设备测试。
- CANN：CCE kernel、ACL host、build、correctness、设备内存评估、benchmark 和 demo 已建立为 scaffold；本仓库尚未确认此算子通过 bisheng 构建或任何 Ascend 设备测试。
- `operator.yaml` 不记录任何新算子的实测性能行；只有在目标环境实际运行后才可更新状态和结果。

## 验收标准

- 每个选定 kernel/host 组合通过对应后端正确性工程，覆盖 `N=0`、单元素、常规长度、非整齐尾部、未对齐指针（CUDA）和非法参数。
- CUDA/CANN 内存评估记录输入规模、三缓冲理论申请量、设备总内存和分配前/后/释放后可用量。
- benchmark 记录设备、源码选择、输入规模、dtype、编译器/runtime、测量延迟、估算有效带宽和结果校验；没有实测时不填性能数据。
- CUDA float32 `multiplication`按每元素 2 次读和 1 次写估算 `12*N` 字节；这是有效带宽估算，不是硬件计数器测得的 DRAM 带宽。
- 仅在实际工具链和硬件验证后，才可把后端 `status` 从 `scaffold` 更新为已验证状态。
