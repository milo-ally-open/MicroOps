# OpsFromScratch

OpsFromScratch 用于从零学习 CUDA 与 CANN 算子开发，并按可测试、可评估、可打包的方式组织设备算子工程。仓库不维护 CPU 算子后端。当前首个算子是 CUDA `add` v1；CANN 目录和测试入口仅为 scaffold。

## 目录概览

- `operators/<op>/cuda/`、`operators/<op>/cann/`：各算子的设备 kernel、host 调度和后端构建配置。
- `operators/<op>/tests/cuda/`、`operators/<op>/tests/cann/`：独立的正确性、内存和性能评估工程；结果保存在对应 suite 的 `results/` 中。
- `operators/<op>/reference/`：论文和参考资料预留目录；当前只包含 `.gitkeep`。
- `operators/<op>/docs/`：算子提案、规格、任务和用法。
- `docs/sop.md`：统一算子开发工作流。
- `tools/`：按算子与后端构建、测试、benchmark、打包的入口。
- `.agents/`、`plugins/`：可跟踪的项目级 Codex 技能和插件产物；`.codex/` 是被忽略的本地配置目录。
- `build/`、`dist/`：本地生成目录，已加入 `.gitignore`。

## 已有算子

| 算子 | CUDA | CANN |
|---|---|---|
| `add` | v1 kernel/host 已存在；历史设备 smoke test 与 benchmark 有记录，新目录布局待复验 | scaffold，尚无实现或测试 harness |

CUDA 状态见 [`operators/add/operator.yaml`](operators/add/operator.yaml)。历史记录不代表当前机器或新测试工程已验证。

## CUDA 构建、测试和 demo

在已安装 CUDA Toolkit、CMake 3.21+ 和 C++17 编译器的 WSL/Linux 环境中：

```bash
python3 tools/build.py --operator add --backend cuda
python3 tools/test.py --operator add --backend cuda
python3 tools/build.py --operator add --backend cuda --examples
./build/add/cuda/add_demo
```

`tools/test.py` 构建并运行 `operators/add/tests/cuda/` 独立测试工程，覆盖正确性和小规模显存评估。没有 CUDA 编译器时会明确报告配置失败；没有设备/驱动时设备用例会 Skip，不计为通过。

显式运行性能评估：

```bash
python3 tools/benchmark.py --operator add \
  --sizes 1024,1048576,16777216 --iterations 100
```

性能测量追加到 `operators/add/tests/cuda/performance/results/cuda.csv`；显存评估记录追加到 `operators/add/tests/cuda/memory/results/memory.csv`。结果记录设备、输入规模、编译器/runtime、资源读数和测量数据；未运行的数据不会生成。

## CANN

`operators/add/cann/` 与 `operators/add/tests/cann/` 当前只有 scaffold。构建和测试入口会明确停止并说明未实现，不会将其误报为可用后端。

## 打包

已实现后端可通过统一入口打包：

```bash
python3 tools/package.py --operator add --backend cuda
```

输出放在 `dist/`。若仓库尚无许可证文件，不应自行添加许可证。

更多细节见 [`docs/build.md`](docs/build.md)、[add 使用说明](operators/add/docs/usage.md) 和[算子开发 SOP](docs/sop.md)。
