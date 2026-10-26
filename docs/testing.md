# 算子测试、bench 与 profile

所有算子和后端统一从仓库根目录运行：

    python3 tools/run_operator.py <test|bench|profile> operators/<op>/operator.yaml --backend <pytorch|triton|cuda|cann>

例如：

    python3 tools/run_operator.py test operators/add/operator.yaml --backend cuda
    python3 tools/run_operator.py bench operators/add/operator.yaml --backend cann
    python3 tools/run_operator.py profile operators/add/operator.yaml --backend pytorch

源文件选择仍通过 YAML 默认值或显式参数完成：

    python3 tools/run_operator.py test operators/add/operator.yaml --backend cuda --kernel add_kernel_v1.cu --host add_host_v1.cu
    python3 tools/run_operator.py bench operators/add/operator.yaml --backend cann --kernel add_kernel_v1.cpp --host add_host_v1.cpp

## 三个阶段的职责

| 命令 | 目的 | 产物 |
|---|---|---|
| test | 构建所选实现，运行正确性和设备内存分配检查；缺少依赖/设备时明确失败或 Skip | 终端测试结果；真实设备内存检查写入 `operators/<op>/tests/<backend>/test/results/` |
| bench | 依照 YAML 中的输入规模、预热与迭代设置测量实际执行时间和有效带宽 | 仅真实运行后写入 `operators/<op>/tests/<backend>/bench/results/` 的 CSV |
| profile | 使用对应后端 profiler 采集 kernel / runtime 活动、设备指标或 trace | 写入被 `.gitignore` 忽略的 `build/<op>/profile/<backend>/` |

CUDA 与 CANN 的 test/bench 由各自独立 CMake 测试工程构建。PyTorch/Triton 的 test/bench 由 Python GPU 实现入口运行。四种后端都必须是目标 GPU/NPU 后端；框架张量不允许 CPU 回退。配置中 `status` 是源码验证状态，只有实际构建、正确性测试和设备测试完成后才可更改。若所选后端标记为 `scaffold`、`not_implemented` 或 `unavailable`，统一入口会在探测工具链前输出 `SKIP`、不生成结果，并以退出码 77 结束；这不是 PASS。依赖缺失或设备不可用则报告错误，不会降级到 CPU。

## Profile 工具与限制

- CUDA 优先使用 NVIDIA Nsight Compute CLI（`ncu`）采集 kernel 计数器。若 `ncu` 未安装/不可见，或因 `ERR_NVGPUCTRPERM` 无法读取计数器，而 Nsight Systems（`nsys`）可用，统一入口会回退为 CUDA API/kernel 时间线采集，并在终端标为 `cuda-timeline`。时间线可以分析 launch、kernel 时长和 memcpy 活动，但不包含 NCU 硬件计数器；不能将二者混称。本轮 WSL 用 Nsight Systems 补齐了九个 CUDA 算子的时间线 profile。
- CANN 使用板端 `msprof` 包装已构建的 CANN bench executable。需先加载匹配版本的 CANN 环境；数据采集本身可能增加执行开销，profile 时间不得当成 bench 延迟。
- PyTorch 与 Triton 使用 `torch.profiler` 采集 CPU 调度、CUDA kernel 活动和显存分配，并导出 Chrome trace、统计摘要和 metadata。Triton 的 JIT 预热在 profile 采集窗口外执行。
- profiler 不可用、设备无法访问、构建失败或采集失败时命令返回失败状态，不创建“通过”或空白的性能结果。profile 参数由每个算子的 `operator.yaml` 中 `profile` 节配置，单次输入规模和迭代数可用命令行覆盖。

对应工具文档：[PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler.html)、[NVIDIA Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/)、[MindStudio 7.0 msprof 采集说明](https://www.hiascend.com/document/detail/en/mindstudio/700/TITools/Profiling/atlasprofiling_16_0011.html)。板端 CANN 7.0.RC1 的 msprof 要求用 `--application=<executable and arguments>` 指定应用；统一脚本会按此形式传入 bench 程序。

## 结果解释

bench CSV 记录设备、输入规模、dtype、所选实现、编译器/运行时版本、迭代数、分配状态、平均延迟与有效带宽。Elementwise 带宽为读写字节数除以实测时间的估算；不是硬件计数器读数。PyTorch/Triton 的 benchmark 包括 wrapper 的输出分配开销；CUDA/CANN bench harness 记录其设备缓冲区分配与执行延迟。比较不同后端时需使用相同设备、输入规模、dtype 和相近的软件环境，不能将 profile 采集期间的数值与 bench CSV 混用。

所有数据必须来自实际运行。未安装后端或没有设备时，不生成或填造性能数字；后端仍标记为 scaffold/source_only，直到完成对应验证。本轮实际运行范围、代表性测量和限制见 [性能分析](performance-analysis.md)。
