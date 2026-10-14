# 构建、测试与评估

算子在 `operators/<op>/` 内独立配置。CUDA、CANN 交付件分别位于该算子的 `src/cuda/` 和 `src/cann/`；对应测试工程位于 `tests/cuda/` 和 `tests/cann/`。本仓库不提供 CPU 算子后端。

## 工具要求

- CMake 3.21+
- Python 3.9+（统一工具脚本）
- CUDA：兼容的 CUDA Toolkit/nvcc、C++17 host 编译器，以及设备测试所需驱动/设备
- CANN：匹配的 CANN 开发环境与 NPU；当前 add 后端和测试工程仍是 scaffold

## CUDA add

在 WSL/Linux 仓库根目录运行：

```bash
python3 tools/ops.py build --operator add --backend cuda
python3 tools/ops.py test --operator add --backend cuda
python3 tools/ops.py build --operator add --backend cuda --examples
```

直接构建算子交付件：

```bash
cmake -S operators/add -B build/add/cuda \
  -DOPS_ENABLE_CUDA=ON -DOPS_ENABLE_CANN=OFF
cmake --build build/add/cuda --config Release
```

测试工程单独位于 `operators/add/tests/cuda/`，构建输出位于 `build/add/tests/cuda/`。它直接引用 `operators/add/src/cuda/` 中的 kernel/host，并运行 correctness 与 memory suites。没有 `nvcc` 时配置明确失败；没有设备或驱动时对应测试明确 Skip。

## 性能评估

性能程序由 CUDA 测试工程构建，测量和记录使用统一入口：

```bash
python3 tools/ops.py benchmark --operator add \
  --sizes 1024,1048576,16777216 --iterations 100
```

结果追加到 `operators/add/tests/cuda/performance/results/cuda.csv`，显存评估结果写入 `operators/add/tests/cuda/memory/results/memory.csv`。新算子必须在自己的测试工程和元数据中登记对应的评估目标。

## CANN 与打包

```bash
python3 tools/ops.py build --operator add --backend cann
python3 tools/ops.py test --operator add --backend cann
```

目前以上 CANN 命令会以明确错误停止，因为 add 尚无 CANN 实现和测试 harness。不要将 scaffold 当作成功构建。CUDA 源码和测试工程可打包至 `dist/`：

```bash
python3 tools/ops.py package --operator add --backend cuda
```

此前在 WSL 上运行过 add CUDA v1 的 smoke test 和多规模性能评估，记录保存在 `operators/add/tests/cuda/performance/results/cuda.csv`。目录布局及独立测试工程调整后，需在 CUDA 工具链可用时重新验证。
