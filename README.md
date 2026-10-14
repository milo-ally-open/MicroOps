# OpsFromScratch

OpsFromScratch 是一个用于学习 CUDA 与 Ascend C 并行计算、同时保持工业算子工程习惯的仓库。每个算子包含可验证的 CPU 参考实现、独立后端入口、测试、示例和可打包说明。当前没有任何 GPU/NPU 后端被本机验证为可用。

## 目录概览

- `operators/`：按算子组织实现、规格、测试、示例、benchmark 和动画源码。
- `docs/`：仓库架构、构建和贡献约定。
- `showcases/`：组合多个算子的应用示例。
- `tests/`：跨算子测试。
- `tools/`：统一构建、测试、benchmark 和打包入口。
- `cmake/`：共享 CMake 配置。
- `build/`、`dist/`：本地构建与打包输出，不纳入版本控制。

## 已有算子

| 算子 | CPU 参考 | CUDA | Ascend C |
|---|---|---|---|
| `reduce_sum` | C++17 实现；本机尚未编译验证 | scaffold，未验证 | scaffold，未验证 |

每个算子的完整定义和验收条件见 `operators/reduce_sum/docs/`。后端状态及入口见 `operators/reduce_sum/operator.yaml`。

## CPU-only 快速开始

需要 CMake 3.21 或更高版本、C++17 编译器，以及运行统一脚本所需的 Python 3.9 或更高版本。默认配置只构建 CPU 参考实现，不查找 CUDA 或 CANN。

```powershell
python tools/build.py --backend cpu
python tools/test.py --backend cpu
```

也可以直接使用 CMake preset：

```powershell
cmake --preset cpu
cmake --build --preset cpu
ctest --preset cpu
```

## 可选后端

在 CUDA Toolkit、支持 CUDA 的 C++ 编译器及设备就绪后，可单独尝试：

```powershell
python tools/build.py --backend cuda
python tools/test.py --backend cuda
```

在安装并配置 CANN/Ascend C 工具链后，可单独尝试：

```powershell
python tools/build.py --backend ascendc
python tools/test.py --backend ascendc
```

两个后端默认关闭，不会影响 CPU-only 配置。当前 Ascend C 入口明确报告 scaffold 尚未实现；CUDA 源码也尚未在工具链和设备上验证。工具链或硬件相关检查失败时，命令会报告原因，不会把跳过的后端测试计为通过。

## Demo、benchmark 与打包

`operators/reduce_sum/examples/grayscale_mean.cpp` 展示如何先逐像素转换灰度，再通过 `reduce_sum` 汇总像素值并计算均值。

```powershell
python tools/build.py --backend cpu --examples
python tools/benchmark.py --backend cpu --sizes 1024,1048576
python tools/package.py --operator reduce_sum --backend cpu
```

Visual Studio 等多配置生成器会将 Release demo 放在 `build/cpu/Release/`；单配置生成器直接放在 `build/cpu/`。

benchmark 默认写入被忽略的 `build/benchmark_results.csv`；只有实际执行后才会包含测量数据。package 输出到 `dist/`，不修改源码目录。
