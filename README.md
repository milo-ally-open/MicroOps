# OpsFromScratch

OpsFromScratch 用于学习 CUDA 与 CANN 并行计算，并以可测试、可评估、可打包的方式组织设备算子工程。仓库不维护 CPU 算子后端。首个算子 `add` 有两组 CUDA kernel/host 示例；CANN 仍为 scaffold。

## 目录概览

- `operators/<op>/src/{cuda,cann}/`：设备 kernel、host 调度、头文件和后端构建配置。
- `operators/<op>/tests/{cuda,cann}/`：独立正确性、内存和性能评估工程及结果。
- `operators/<op>/docs/`：算子提案、规格、任务、用法、学习笔记和论文参考资料。
- `operators/<op>/examples/`、`operators/<op>/animation/`：小型 demo 和动画源码。
- `docs/sop.md`：通用算子开发工作流。
- `tools/ops.py`：统一的构建、测试、性能评估和打包入口。
- `.agents/`、`plugins/`：可跟踪的项目级 Codex 技能和插件产物；`.codex/` 是被忽略的本地配置目录。
- `build/`、`dist/`：本地生成目录，已加入 `.gitignore`。

## 已有算子

| 算子 | CUDA | CANN |
|---|---|---|
| `add` | v1/v2 已在 RTX 4060 Laptop GPU 上构建、测试和测量 | scaffold，无实现 |

每个算子的 `operator.yaml` 是构建工具配置入口，记录后端源码目录、默认 kernel/host 文件、测试工程和结果路径。查看配置中的可选源文件：

```bash
python3 tools/ops.py list operators/add/operator.yaml
```

## CUDA 快速开始

安装 Python 配置解析依赖，再从仓库根目录执行：

```bash
python3 -m pip install -r requirements.txt
python3 tools/ops.py build operators/add/operator.yaml --examples
python3 tools/ops.py test operators/add/operator.yaml
./build/add/cuda/add_kernel_v2__add_host_v2/add_demo
```

当前 `operator.yaml` 默认选择 `add_kernel_v2.cu` 与 `add_host_v2.cu`。切换到 v1 示例时，可在配置中修改 `default_file`，或临时传 `--kernel add_kernel_v1.cu --host add_host_v1.cu`。文件名只是选择值，不要求采用 v1/v2 格式。

CUDA 构建需要 CUDA Toolkit/nvcc、CMake 3.21+ 和 C++17 host 编译器。没有 CUDA 编译器时配置会明确失败；没有设备或驱动时设备用例会 Skip，不计为通过。当前 CUDA 验证使用 nvcc 12.6、CUDA Runtime 12.6 和 RTX 4060 Laptop GPU；v1/v2 的 correctness、memory 和 benchmark 记录位于各自测试结果目录。单轮 benchmark 记录用于观察，不代表普遍性能结论。

## 性能评估与打包

正确性通过后执行：

```bash
python3 tools/ops.py benchmark operators/add/operator.yaml \
  --sizes 1024,1048576,16777216 --iterations 100
python3 tools/ops.py package operators/add/operator.yaml
```

评估记录设备、源码文件、输入规模、编译器/runtime、内存数据和测量结果；未运行的数据不会生成。源码包输出到 `dist/`，不代表已构建或通过设备验证。

## CANN

`operators/add/src/cann/` 和 `operators/add/tests/cann/` 当前只有 scaffold。工具可按 `operator.yaml` 配置读取 CANN 文件路径；当前没有可构建的 add CANN 实现。

更多说明见 [`docs/build.md`](docs/build.md)、[add 使用说明](operators/add/docs/usage.md) 和[算子开发 SOP](docs/sop.md)。若仓库尚无许可证，不应自行添加许可证。
