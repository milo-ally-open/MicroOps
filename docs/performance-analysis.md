# Add–Tanh 性能分析与本轮优化

本报告汇总 2026-10-14 至 2026-10-26 在实际设备上保存的 test、bench 与 profile 证据。记录按每个 CSV 中最新的对应规模行读取；这些运行不是严格同一时刻的受控对照，GPU 当时还有其他进程负载，因此适合定位主要瓶颈，不应用来宣布细微差异或跨设备排名。

## 验证环境与范围

- GPU：NVIDIA GeForce RTX 4060 Laptop GPU；WSL2；nvcc 12.6.85、CUDA Runtime 12.6、驱动 API 12.7；PyTorch 2.14.1+cu126、Triton 3.8.0。
- NPU：Atlas 200I DK A2 / Ascend 310B4；CANN 7.0.RC1；bisheng clang 15.0.5；目标 dav-m300；平台配置报告 `ai_core_cnt=1`。
- GPU 的 16M 行按 float32 elementwise 的配置规模记录；CANN 的代表规模为 1M。effective bandwidth 是基于逻辑读写字节和测得时间计算的估算值，不是硬件 DRAM counter。
- GPU 测量期间可见约 39% 的 GPU 利用率和约 1.5 GiB 显存占用。Nsight Compute 退出 `ERR_NVGPUCTRPERM`，所以这轮没有取得 CUDA 硬件计数器数据。PyTorch/Triton 的 `torch.profiler` trace 已采集；CANN 用 msprof 采集，报告包含 task activity，但部分汇总指标显示 N/A。

### CUDA 时间线 profile（Nsight Systems）

由于当前 WSL 会话不能访问 NCU 硬件计数器，安装用户级 Nsight Systems CLI 2026.5.1 后，对九个 CUDA YAML 默认版本分别采集了 `N=65,536` 的 CUDA API/kernel 时间线。以下为 Nsight Systems `cuda_gpu_kern_sum` 中 kernel 设备执行时间中位数；每个 harness 样本包含 11 次 kernel launch。profile 启动开销、系统其他 GPU 进程及采集插桩会影响数值，故此表只用于确认活动和粗略排查，不用于替代 bench 排名或硬件计数器分析。

| 算子 | CUDA 默认版本 | kernel 中位时间（µs） |
|---|---|---:|
| add | v2 | 2.371 |
| sub | v2 | 2.281 |
| mul | v2 | 2.255 |
| div | v2 | 3.089 |
| relu | v2 | 1.924 |
| gelu | v2 | 2.567 |
| silu | v2 | 2.890 |
| sigmoid | v2 | 2.921 |
| tanh | v2 | 2.507 |

时间线同时确认了 host 侧的 CUDA API、输入输出拷贝和 kernel launch；短小 profile harness 中 `cudaMemGetInfo` 的 host API 时间占比较高，说明这类小规模场景容易受管理调用与 launch 开销影响。Nsight Systems 时间线没有 DRAM 带宽、SM/SFU 利用率、occupancy 或寄存器压力计数器；Add/Sub/Mul 的内存受限判断仍来自读写字节模型及较大规模 bench，激活算子的算术/超越函数瓶颈也不能仅凭本次时间线定量确认。原始 `.nsys-rep` 与 `.sqlite` 文件保存在忽略的 `build/<op>/profile/cuda/` 下。

## GPU 大规模 benchmark 摘要

以下为 N=16,777,216 的最新保存延迟，单位为微秒。每个实现的原始数据（包括设备、软件版本、迭代数、分配信息和有效带宽）在对应 `operators/<op>/tests/<backend>/bench/results/` CSV 中。

| 算子 | PyTorch | Triton | CUDA 当前 YAML 版本 |
|---|---:|---:|---:|
| add | 1591.5 | 1422.8 | v2 1400.6 |
| sub | 1326.5 | 1395.4 | v2 1367.7 |
| mul | 1336.9 | 1350.2 | v2 1377.7 |
| div | 1366.7 | 1296.5 | v2 1487.5 |
| relu | 918.1 | 975.4* | v2 954.5 |
| gelu | 923.3 | 906.2 | v2 841.4 |
| silu | 943.1 | 923.3 | v2 984.0 |
| sigmoid | 912.7 | 903.6 | v2 910.1 |
| tanh | 897.9 | 1097.5 | v2 912.3 |

`*` ReLU Triton v2 候选在同规模为 975.4 us，慢于 v1 的 905.8 us，因此 YAML 仍选择 v1；该版本作为已测量的候选保留，不称为优化。CUDA v2 是此前建立的向量化/多元素处理候选，不应默认理解为一定快于 v1；这轮配对测量中多数差值小且受负载影响，Div、SiLU 等行还出现退化。

## 内存评估

这些测试测量设备缓冲区能否分配、申请前后的可用显存/设备内存，以及框架 profile 暴露的峰值分配；它们不是显存控制器的物理流量计数器。对连续 float32 的理论工作集：Add/Sub/Mul/Div 是两个输入加一个输出，约 `12*N` bytes；激活算子是一个输入加一个输出，约 `8*N` bytes。N=16,777,216 时分别为 192 MiB 与 128 MiB；N=1,048,576 时分别为 12 MiB 与 8 MiB。

各 CUDA/CANN memory.csv 记录每个后端实际设备总量、分配前/后/释放后的空闲内存和分配结果；bench CSV 同时记录该 bench harness 的工作集和设备内存状态。PyTorch profiler 显示旧 ReLU 表达式有额外中间张量；换成 `torch.relu` 后临时分配显著收敛，峰值接近输入与输出缓冲区。其他实现也会因框架分配器缓存而使进程可见占用与单次算子理论工作集不同，不能用系统级 `nvidia-smi` 数字代替算子分配记录。

## CANN 大规模 benchmark 摘要

N=1,048,576，2026-10-26 在同一板端环境分别用 v1、v2 各测 30 次所得的延迟如下。单位为毫秒；本轮每个版本记录 3 个实际尺寸，原始设备、编译器、运行时、内存与带宽估算见 `operators/<op>/tests/cann/bench/results/` 中对应版本 CSV。

| 算子 | CANN v1 | CANN v2 | 同组差异与判断 |
|---|---:|---:|---|
| add | 46.141 ms | 41.838 ms | v2 低约 9.3%；仍只有约 0.27–0.30 GB/s 的估算有效带宽 |
| sub | 46.194 ms | 42.298 ms | v2 低约 8.4%；瓶颈仍是标量 GM 读写/并行度 |
| mul | 46.135 ms | 41.838 ms | v2 低约 9.3%；当前实现未使用经验证的向量数据通路 |
| div | 63.080 ms | 65.447 ms | v2 高约 3.8%；四元素展开没有带来改善 |
| relu | 41.442 ms | 37.505 ms | v2 低约 9.5%；仍受标量 GM 读写与并行度限制 |

本组 CANN v1/v2 都通过 correctness 与设备内存评估；每版各有 3 个真实 bench 尺寸。`msprof` 对 10 个组合均采集到 profile timeline，摘要中的 top-time/counter 字段仍为 N/A，因此 profile 可用于确认设备任务活动，不能拿来宣称精确硬件吞吐。profile 原始文件在被忽略的 `build/<op>/profile/cann/` 下。

对 `msprof` 原始 `task_time` 中 `KERNEL_AICORE` 项按 65536 元素 profile 的样本取中位数，可解析到以下设备执行时间（微秒）。算术算子各 4 个样本，ReLU 各 22 个样本；样本量有限，适合和 1M bench 的方向相互印证，不作为精细调优排名。

| 算子 | CANN v1 | CANN v2 |
|---|---:|---:|
| add | 2750.92 | 2468.76 |
| sub | 2748.92 | 2465.88 |
| mul | 2764.19 | 2467.24 |
| div | 3764.68 | 3847.19 |
| relu | 2340.86 | 2116.52 |

这套板卡安装的 CANN 7.0.RC1、bisheng clang 15.0.5 与 dav-m300 目标组合下，Ascend C `LocalTensor`/`TPipe` vector API 探针未能通过编译，诊断包含头文件中的 `HardEvent` 解析失败及缺失设备实现符号；安装包的 `tikcfw/impl` 也只有 `dav_c100`、`dav_m200`、`dav_c220` 分支，没有 dav-m300 实现。标量 `expf`/`tanhf` 会被判为 host 函数，改成 compiler builtin 后又留下设备链接未解析符号。额外测试的 CCE `vexp` 向量 builtin 在该目标上触发 bisheng backend lowering 错误。因此 GELU、SiLU、Sigmoid、Tanh 的 CANN 继续标记为 `scaffold`，没有 CANN 性能数据。CANN 7.0.RC1 的官方 Ascend C 指南确实列出了支持 Atlas 推理系列 AI Core 的 `Exp` 向量 API，但这不足以证明当前安装包的 dav-m300 实现可用；应以目标板实编译和设备正确性测试为准。后续需先安装或匹配包含 dav-m300 实现的 SDK，再编译最小 `Exp`/`Tanh` probe，完成 kernel、test、bench 和 profile，不能仅凭新版 API 文档把状态升级。

官方版本参考：[CANN 7.0.RC1 Ascend C 算子开发指南](https://www.hiascend.com/doc_center/source/zh/canncommercial/70RC1/operatordev/Ascendcopdevg/CANN%207.0.RC1%20Ascend%20C%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E6%8C%87%E5%8D%97%2001.pdf)。

### 2026-10-26 开发板复核

通过 Windows 主机连接 Atlas 200I DK A2 后确认 NPU 为 310B4、状态 OK，`npu-smi` 报告驱动工具版本 23.0.RC3；`bisheng --version` 报告 clang 15.0.5。为保护板端仓库未提交改动，验证与本轮 9 算子 CANN 工程均在 `/tmp/OpsFromScratch-full-CANN-validation` 隔离副本运行。add/sub/mul/div/relu 的 v1、v2 均通过 CANN correctness、设备内存评估、3 尺寸 bench 与 `msprof` profile。四个激活算子的 CANN `test` 构建均未通过：标准 `expf`/`tanhf` 被解析为 host 函数；改写成 `__builtin_expf`/`__builtin_tanhf` 后虽能编译，却分别在设备链接阶段留下未解析的 `expf`/`tanhf` 符号。向量 API 探针也失败：安装的 tikcpp 实现目录只有 `dav_c100`、`dav_m200`、`dav_c220` 分支，无法匹配编译器目标 `dav-m300`，出现 CCE intrinsic 签名及架构定义错误。故这轮没有把激活 CANN 标为可用，也没有记录 bench/profile 数据；应先安装一套编译器、运行时、设备支持和 Ascend C API 版本相互匹配的 SDK。

## 瓶颈判断

1. **CUDA/PyTorch/Triton 上的 Add/Sub/Mul 是内存流量主导。** 每个 float32 输出至少需要读取两个输入并写一个输出，即约 12 bytes/element。大输入运行时间落在约 1.3–1.6 ms 区间，达到约 126–152 GB/s 的估算有效带宽量级；增加每线程工作量或展开并未稳定提高结果。
2. **ReLU PyTorch 初始实现多次启动并产生临时张量。** profiler 显示旧表达式由比较、NaN 检查、填充和 where 等多个 CUDA 操作组成；大规模旧测量约 4.17 ms。改为 `torch.relu(x)` 后 trace 收敛为单个算子路径，实测 v2 约 0.918 ms，显存峰值接近输入/输出各一份的必要量，明显减少了多 kernel 与临时内存成本。
3. **激活 CUDA kernel 已接近“单 kernel + 读写”基线。** GELU/SiLU/Sigmoid/Tanh 包含超越函数或乘加，计算量高于 ReLU，但当前单 kernel 时间与其他 GPU 实现同量级。没有 Nsight 计数器，不能可靠区分 SFU 吞吐、内存或占用率的贡献；CUDA v2 的多元素路径也未呈现普遍稳定增益。
4. **Triton 的块尺寸变化没有稳定收益。** 对多个 block size 的临时 sweep 重复后观察到噪声；官方 wrapper 上 ReLU v2 还比 v1 慢约 7.7%。因此保留 v1 默认，不把单次微基准波动当作优化证据。
5. **当前 CANN elementwise 实现受到标量访存和 launch/grid 策略约束。** v1/v2 在已测 310B4 上以单 block 标量循环运行；四元素手工展开只增加指令级并行尝试，不代表 SIMD。当前 SDK/目标组合无法验证向量 API，故不做伪优化或推测吞吐。

## 已做的优化与后续方向

- ReLU PyTorch 新增 `relu_pytorch_v2.py` 并将 YAML 默认实现切到 `torch.relu`。NaN 测试通过；profile 和 bench 证据对应实际 GPU 运行。
- ReLU CUDA/CANN kernel 将依赖 `x != x` 的 NaN 保护改为有序比较 `x < 0 ? 0 : x`。NaN 条件为假，仍传递原值，同时消除目标 CANN 编译器不支持的浮点自比较；两个版本均重跑设备正确性/内存检查。该改动主要解决可构建性和指令简化，不能声称已有稳定加速。
- ReLU Triton v2 作为 1024 block 候选完成 test、bench 和 profile，但实测慢于 v1，默认仍为 v1。
- 对其余算子不因一次性轻微波动改写 kernel。下一轮有价值的实验是：在空闲 GPU 上重复、随机交替版本并报告中位数/离散度；获得 Nsight 许可后分析实际内存带宽、SM/SFU 吞吐、占用和寄存器；CANN 匹配可用的 m300 SDK 后重做 vector 路径。

## 状态与数据位置

`operator.yaml` 的 `device_tested` 表示该设备上的构建/正确性/benchmark 验证，不保证硬件 counter profile 成功。CUDA、PyTorch、Triton 九个算子的设备 test/bench/profile 已执行；CUDA 使用 Nsight Systems 时间线 profile，NCU counters 因权限限制未完成，PyTorch/Triton profile 使用 `torch.profiler`。CANN 的 Add/Sub/Mul/Div/ReLU v1/v2 已在板端 test/bench/profile；其他四个激活函数仍为 scaffold。数据和 profile 的具体路径见各自 `tests/<backend>/{test,bench}` 目录及被忽略的 `build/<op>/profile/<backend>/`。
