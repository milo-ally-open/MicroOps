# sigmoid 规格

## 定义

对输入向量 `X` 的每个元素独立计算：`y = 1/(1+exp(-x)); use sign-dependent forms to avoid overflow.` 输出 `Y[i]`。

## 输入与输出

- `X`：连续 float32 设备输入；`Y`：至少容纳 N 个 float32 的设备输出；`N` 为元素数。
- 接口支持 `0 <= N <= INT_MAX`。`N == 0` 不访问指针且不启动 kernel；`N > 0` 时指针不可为空。
- host 对负长度或非零长度空指针返回后端非法参数错误；调用者负责分配、传输和同步。
- 不支持 stride、广播或破坏读取的输入输出重叠别名。

## dtype、shape 与版本

- 当前仅支持 float32，逻辑 shape 为 `[N]`。
- v1：一个线程处理一个元素。
- v2：每线程按四元素间隔展开并保护尾部；这是实现候选，不代表经测量的优化。
- CUDA 与 CANN 状态均为 scaffold，直到相应工具链构建、正确性测试和设备验证完成。

## 数值精度与边界

- 数学定义：y = 1/(1+exp(-x)); use sign-dependent forms to avoid overflow.
- 验收：abs(actual - expected) <= 2e-6 for finite float32 inputs。
- 测试覆盖 N=0、1、259、1027、负 N/空指针错误，以及正负输入、零点和饱和区样本；CUDA correctness 另用偏移设备指针检查对齐敏感路径。
- correctness 用例以有限 float32 为主。NaN/Inf 的跨后端一致性暂不承诺；设备数学函数边界行为需在对应工具链上验证。

## 后端状态

- CUDA：v1/v2 源码和构建/测试入口已建立；本次初始化未在 CUDA 工具链/设备上验证，YAML 标记 scaffold。
- CANN：v1/v2 CCE/host 源码 scaffold 和构建/测试入口已建立；本次初始化未在 CANN 工具链/设备上验证，YAML 标记 scaffold。
- 任何性能 CSV 只有实际运行后才写入；理论工作集为每元素 8 bytes（一次输入读取、一次输出写入），有效带宽为估算值。

## 验收标准

- 两组兼容 kernel/host 组合分别通过 correctness 测试，含尾部、未对齐输入和非法参数。
- 内存测试记录两个 float32 缓冲区的工作集、设备内存与分配状态。
- benchmark 记录设备、源码选择、输入规模、编译器/runtime、延迟、估算带宽和结果校验；未实测不填数值。
