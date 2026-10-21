# mul 使用说明

## 配置

`operator.yaml` 是构建、测试、benchmark 和打包入口。CUDA 默认选择 `mul_kernel_v2.cu` / `mul_host_v2.cu`，CANN 默认选择 `mul_kernel_v1.cpp` / `mul_host_v1.cpp`。文件名只是源码选择；可继续新增 v3/v4 并通过 `--kernel`、`--host` 指定。

注意：此算子的 CUDA/CANN 当前均为 scaffold。下面命令是验证完成后使用的入口；命令能启动不代表后端已在当前机器或设备上验证。

## CUDA

需 CUDA Toolkit/nvcc、CMake 3.21+、C++17 host 编译器和 CUDA GPU：

```bash
python3 tools/operator_cuda.py build operators/mul/operator.yaml --examples
python3 tools/operator_cuda.py test operators/mul/operator.yaml
python3 tools/operator_cuda.py benchmark operators/mul/operator.yaml
./build/mul/cuda/mul_kernel_v2__mul_host_v2/mul_demo
```

选择 v1：

```bash
python3 tools/operator_cuda.py build operators/mul/operator.yaml --kernel mul_kernel_v1.cu --host mul_host_v1.cu --examples
python3 tools/operator_cuda.py test operators/mul/operator.yaml --kernel mul_kernel_v1.cu --host mul_host_v1.cu
python3 tools/operator_cuda.py benchmark operators/mul/operator.yaml --kernel mul_kernel_v1.cu --host mul_host_v1.cu
```

## CANN

需目标机 CANN Toolkit、bisheng、ACL Runtime、CMake 3.21+ 和可用 Ascend 设备；先加载对应版本环境：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 tools/operator_cann.py build operators/mul/operator.yaml --examples
python3 tools/operator_cann.py test operators/mul/operator.yaml
python3 tools/operator_cann.py benchmark operators/mul/operator.yaml
./build/mul/cann/mul_kernel_v1__mul_host_v1/mul_demo
```

CANN 文件从 `add` 的 CCE/ACL scaffold 结构派生，CMake 当前目标架构沿用 `dav-m300`；在确认实际板卡和 SDK 兼容前，不得据此声称已支持该设备。ACL host 接收设备指针和 stream，由调用方负责内存生命周期与同步；测试不依赖 PyTorch。

## 测试、评估和打包

正确性/内存测试位于 `tests/<backend>/`，benchmark 规模和迭代次数来自本文件。逐元素 `multiplication`每个元素按两次读、一次写估算 working set `12*N` 字节；带宽是事件计时推算值，不是硬件计数器测量。测试或 benchmark 未实际运行时不会生成结果记录。

```bash
python3 tools/operator_cuda.py package operators/mul/operator.yaml
python3 tools/operator_cann.py package operators/mul/operator.yaml
```

包输出在 `dist/`，其源码包不代表目标后端已构建或经过设备验证。
