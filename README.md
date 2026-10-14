# OpsFromScratch

OpsFromScratch 用于学习 CUDA 与 CANN 并行计算，并以可测试、可评估、可打包的方式组织设备算子工程。仓库不维护 CPU 算子后端。当前包含 add：CUDA v1/v2 示例，以及已在 Atlas 200I DK A2 / Ascend 310B4 上用 CANN 7.0 RC1 实测的 CANN v1。

## 目录概览

- operators/<op>/src/{cuda,cann}/：设备 kernel、host 调度、头文件和后端构建配置。
- operators/<op>/tests/{cuda,cann}/：正确性、设备内存和性能评估工程及真实运行记录。
- operators/<op>/docs/：该算子的提案、规格、任务和用法文档。
- operators/<op>/examples/、operators/<op>/animation/：小型 demo 和动画源码。
- docs/sop.md：通用算子开发工作流。
- tools/ops.py：统一的构建、测试、性能评估和打包入口。
- .agents/、plugins/：可跟踪的项目级 Codex 技能和插件产物；.codex/ 是被忽略的本地配置目录。
- build/、dist/：本地生成目录，已加入 .gitignore。

## 已有算子

| 算子 | CUDA | CANN |
|---|---|---|
| add | v1/v2 已在 RTX 4060 Laptop GPU 上构建、测试和测量 | v1 已在 Atlas 200I DK A2 / Ascend 310B4、CANN 7.0 RC1 上构建、测试和测量 |

查看 YAML 配置中的可选源码文件：

    python3 -m pip install -r requirements.txt
    python3 tools/ops.py list operators/add/operator.yaml

## CPU-only 快速开始

CPU-only 主机可查看和打包源码；不会因此安装或调用 CUDA/CANN 工具链。仓库当前没有 CPU 算子执行后端，因此设备测试需对应 SDK 与硬件。

    python3 -m pip install -r requirements.txt
    python3 tools/ops.py list operators/add/operator.yaml
    python3 tools/ops.py package operators/add/operator.yaml

## CUDA 环境

需要 CUDA Toolkit/nvcc、CMake 3.21+ 和 C++17 host 编译器：

    python3 tools/ops.py build operators/add/operator.yaml --backend cuda --examples
    python3 tools/ops.py test operators/add/operator.yaml --backend cuda
    python3 tools/ops.py benchmark operators/add/operator.yaml --backend cuda
    ./build/add/cuda/add_kernel_v2__add_host_v2/add_demo

## CANN 环境

需要目标机上的 CANN toolkit、bisheng、ACL runtime、CMake 3.21+ 和可用 Ascend 设备。在 CANN 环境加载后端环境脚本，例如 source <CANN>/set_env.sh：

    python3 tools/ops.py build operators/add/operator.yaml --backend cann --examples
    python3 tools/ops.py test operators/add/operator.yaml --backend cann
    python3 tools/ops.py benchmark operators/add/operator.yaml --backend cann
    ./build/add/cann/add_kernel_v1__add_host_v1/add_demo

CANN v1 当前针对板端 dav-m300，使用 CANN 7.0 RC1 可编译的标量 CCE kernel 与 ACL host 调度；它未在其他 SoC/SDK 组合上验证。具体设备和实测结果写在 operators/add/operator.yaml 与 operators/add/tests/cann/ 的记录中。没有 toolkit/device 时构建或设备测试会明确失败/Skip，不把它们算作通过。

Benchmark 规模与迭代次数来自 operator.yaml，报告记录设备、源码选择、输入规模、编译器/runtime、显存读数、延迟和估算带宽；未实测的数据不会填入。

更多信息见 docs/build.md、operators/add/docs/usage.md 和 docs/sop.md。仓库当前未设置许可证，因此没有添加 LICENSE。
