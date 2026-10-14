# add 规格

## 定义

给定长度为 `N` 的连续向量 `A`、`B`，输出满足：

`C[i] = A[i] + B[i]`, `0 <= i < N`

## 输入与输出

- `A`、`B`：连续 float32 设备输入指针。
- `C`：至少容纳 `N` 个 float32 的设备输出指针。
- `N`：元素数。CUDA v1 使用非负 `int`，因此最大为 `INT_MAX`。
- `N == 0` 时不访问指针、不启动 kernel；`N > 0` 时任一必需指针为空属于调用错误。
- CUDA wrapper 对负数长度或非空长度的空指针返回 `cudaErrorInvalidValue`。

## Dtype、shape 与别名

- 第一版仅支持 float32。
- 逻辑 shape 为 `[N]`；多维数据由调用方展平。
- 不支持 stride 和广播。输出不得与输入发生破坏读取的重叠别名。

## 数值精度

每个输出元素执行一次 IEEE 754 float32 加法。NaN、Inf 和有符号零遵循设备浮点语义。普通有限值以精确可表示用例或 `abs(actual - expected) <= 1e-6` 验收。

## 后端状态

- CUDA：v1 kernel 和 host wrapper 曾在 RTX 4060 上通过 smoke test。源文件已改为算子根目录布局，当前 CMake/独立测试工程尚待重新验证，状态为 `device_tested_cmake_unverified`。
- CANN：尚无实现或测试 harness，状态为 `scaffold`；该状态不代表可构建或可运行。

## 验收标准

- CUDA 正确性测试覆盖空输入、单元素、普通数据、非 block 对齐长度、负数长度和非法空指针。
- CUDA 内存评估记录输入规模、理论申请量、设备总显存、分配前/后/释放后可用量和申请状态。
- CUDA 性能评估记录设备、输入规模、dtype、编译器/runtime/driver、迭代数、延迟、有效带宽估算及结果校验状态。
- 缺少工具链时配置明确失败；缺少设备时设备测试明确 Skip，不能计为通过。
- CANN 仅在实现、测试工程和目标设备验证完成后才可改为可用状态。
