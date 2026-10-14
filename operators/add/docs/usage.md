# add 使用说明

## CUDA v1

CUDA kernel 位于 `cuda/kernel/add_kernel.cu`，host wrapper 位于 `cuda/host/add_cuda.cu`。每线程计算 `idx = blockIdx.x * blockDim.x + threadIdx.x`，wrapper 每 block 使用 256 个线程，并通过 `idx < N` 保护尾部。接口接受设备指针和非负 `N <= INT_MAX`，在指定 stream 异步发射。

CUDA Toolkit、CMake 3.21+ 和 C++17 编译器就绪后，在 WSL/Linux 仓库根目录构建：

```bash
python3 tools/build.py --operator add --backend cuda
python3 tools/build.py --operator add --backend cuda --examples
./build/add/cuda/add_demo
```

直接使用 CMake 时：

```bash
cmake -S operators/add -B build/add/cuda \
  -DOPS_ENABLE_CUDA=ON -DOPS_ENABLE_CANN=OFF
cmake --build build/add/cuda --config Release
```

## CUDA 测试工程

正确性与资源评估使用独立工程 `tests/cuda/CMakeLists.txt`，它直接编译 `cuda/` 中的算子交付件：

```bash
python3 tools/test.py --operator add --backend cuda
```

正确性用例包括 `N=0`、`N=1`、非 block 对齐长度 `N=259`、负数长度和非法空指针。Memory suite 运行小规模三缓冲申请/释放评估，并将实测读数追加至 `benchmarks/cuda/results/memory.csv`。没有可用设备或驱动时，设备测试会显示 Skip；Skip 不代表通过。

CUDA 性能评估在 correctness 和 memory suites 之后显式运行：

```bash
python3 tools/benchmark.py --operator add \
  --sizes 1024,1048576,16777216 --iterations 100
```

结果追加至 `benchmarks/cuda/results/cuda.csv`。每行包含设备、输入规模、编译器/runtime/driver、显存读数、分配状态、延迟、估算有效带宽及校验状态。流量按 A/B 读取和 C 写入估算为 `12 * N` 字节；它不是硬件计数器测得的 DRAM 带宽。

## CANN 状态

`cann/` 和 `tests/cann/` 目前是 scaffold，没有可运行实现。构建或测试命令会明确停止并报告尚未实现，不会回退到其他后端。

## 打包

当前只有 CUDA 后端可打包：

```bash
python3 tools/package.py --operator add --backend cuda
```

包输出至 `dist/`，包括 CUDA kernel/host、独立测试工程、规格和使用说明。当前状态为 CMake/测试工程待复验，源码包不表示已在当前机器构建或通过设备测试。
