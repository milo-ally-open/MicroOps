# MicroOps

MicroOps 是一个面向 AI Infra 学习与实践的算子仓库，用于探索 CUDA 与 CANN 并行计算，并以可测试、可评估、可打包的方式组织设备算子工程。仓库不维护 CPU 算子后端。当前 `add` 至 `tanh` 共九个算子已有 CUDA、CANN、PyTorch 与 Triton 实现入口；各后端验证范围见下表和对应 `operator.yaml`。

## 目录概览

- operators/<op>/src/{cuda,cann,pytorch,triton}/：设备 kernel、host 调度及框架实现；PyTorch/Triton 实现为 CUDA GPU-only 源码。
- operators/<op>/tests/{pytorch,triton,cuda,cann}/{test,bench,profile}/：四个后端的统一测试阶段；CSV 实测记录在 bench/results。
- operators/<op>/docs/：该算子的提案、规格、任务和用法文档。
- operators/<op>/examples/、operators/<op>/animation/：小型 demo 和动画源码。
- docs/sop.md：通用算子开发工作流。
- tools/run_operator.py：PyTorch、Triton、CUDA、CANN 共用的 test/bench/profile 入口；tools/operator_cuda.py 和 tools/operator_cann.py 保留各自的 build/package 入口，共享底层逻辑位于 tools/operator_common.py。
- skills/、plugins/：可跟踪的项目级 Codex 技能和插件产物；.codex/ 是被忽略的本地配置目录副本。
- build/、dist/：本地生成目录，已加入 .gitignore。

## 已有算子

| 算子 | CUDA | CANN | PyTorch | Triton |
|---|---|---|---|---|
| add | device_tested | device_tested | device_tested | device_tested |
| sub | device_tested | device_tested | device_tested | device_tested |
| mul | device_tested | device_tested | device_tested | device_tested |
| div | device_tested | device_tested | device_tested | device_tested |
| relu | device_tested | device_tested | device_tested | device_tested |
| gelu | device_tested | scaffold | device_tested | device_tested |
| silu | device_tested | scaffold | device_tested | device_tested |
| sigmoid | device_tested | scaffold | device_tested | device_tested |
| tanh | device_tested | scaffold | device_tested | device_tested |

`device_tested` 表示相应设备上已完成构建、正确性与设备内存测试，并有实际 benchmark 记录。CANN 激活函数 GELU、SiLU、Sigmoid、Tanh 仍为 scaffold：板端 CANN 7.0.RC1 与 Ascend C 设备数学 API 的目标架构不匹配，未伪报测试通过。CUDA Nsight Compute 硬件计数器受 WSL 驱动权限阻止，性能判断仅使用 benchmark、PyTorch/Triton profiler、CANN msprof 和源码分析；参见 [测试与分析](docs/testing.md)。

查看 YAML 中的源码选择与后端状态：

    python3 -m pip install -r requirements.txt
    less operators/add/operator.yaml

## PyTorch 与 Triton GPU 实现

仓库中的每个算子都提供 PyTorch 和 Triton GPU-only 实现，代码位于各自的 `src/pytorch/` 与 `src/triton/`。这些实现已在 RTX 4060 Laptop GPU 上完成 test、bench 和 `torch.profiler` profile。执行需要 CUDA-enabled PyTorch 和可用 GPU；Triton 版本还需 Triton。CPU 张量不会触发回退。

    python3 tools/run_operator.py test operators/add/operator.yaml --backend pytorch
    python3 tools/run_operator.py test operators/add/operator.yaml --backend triton
    python3 tools/run_operator.py bench operators/add/operator.yaml --backend pytorch
    python3 tools/run_operator.py bench operators/add/operator.yaml --backend triton
    python3 tools/run_operator.py profile operators/add/operator.yaml --backend pytorch
    python3 tools/run_operator.py profile operators/add/operator.yaml --backend triton

bench 默认规模、预热和迭代次数记录在对应 `operator.yaml` 中；实际测量才会追加 CSV 到 `operators/<op>/tests/<backend>/bench/results/`。四套实现共用 `test`、`bench`、`profile` 三个入口，详见 [docs/testing.md](docs/testing.md)。

## CUDA 环境

需要 CUDA Toolkit/nvcc、CMake 3.21+ 和 C++17 host 编译器：

    python3 tools/operator_cuda.py build operators/add/operator.yaml --examples
    python3 tools/run_operator.py test operators/add/operator.yaml --backend cuda
    python3 tools/run_operator.py bench operators/add/operator.yaml --backend cuda
    python3 tools/run_operator.py profile operators/add/operator.yaml --backend cuda
    ./build/add/cuda/add_kernel_v2__add_host_v2/add_demo

CUDA 构建仍使用该算子的 `operator_cuda.py build`；所有后端的 test/bench/profile 使用上面的统一 `run_operator.py` 入口。所有算子的 CUDA 实现均已在 RTX 4060 Laptop GPU 上完成设备 test/bench；CUDA 硬件计数器 profile 受 WSL 权限限制。

## CANN 环境

需要目标机上的 CANN toolkit、bisheng、ACL runtime、CMake 3.21+ 和可用 Ascend 设备。在 CANN 环境加载后端环境脚本，例如 source <CANN>/set_env.sh：

    python3 tools/operator_cann.py build operators/add/operator.yaml --examples
    python3 tools/run_operator.py test operators/add/operator.yaml --backend cann
    python3 tools/run_operator.py bench operators/add/operator.yaml --backend cann
    python3 tools/run_operator.py profile operators/add/operator.yaml --backend cann
    ./build/add/cann/add_kernel_v1__add_host_v1/add_demo

对其他算子使用相同的命令并替换对应 `operator.yaml`。Add/Sub/Mul/Div/ReLU 的 CANN v1/v2 已在 Atlas 200I DK A2（310B4，CANN 7.0.RC1）测试；GELU/SiLU/Sigmoid/Tanh 的 CANN 后端仍是 scaffold。

CANN v1 是单元素标量循环，v2 每轮展开处理四个元素；两者均使用 ACL host 调度并针对 dav-m300。v2 是指令级并行候选，不声称使用 SIMD 指令或在其他 SoC/SDK 上可用。具体设备和实测结果写在 operators/add/operator.yaml 与 operators/add/tests/cann/ 的记录中。没有 toolkit/device 时构建或设备测试会明确失败/Skip，不把它们算作通过。

Benchmark 规模与迭代次数来自 operator.yaml，报告记录设备、源码选择、输入规模、编译器/runtime、显存读数、延迟和估算带宽；未实测的数据不会填入。当前各算子的实测摘要、瓶颈判断和 profile 限制见 [性能分析](docs/performance-analysis.md)。

更多信息见 docs/build.md、operators/add/docs/usage.md 和 docs/sop.md。仓库当前未设置许可证，因此没有添加 LICENSE。
