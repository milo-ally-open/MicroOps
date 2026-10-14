# 构建、测试与评估

每个算子的 `operator.yaml` 集中记录项目根目录、后端源码目录、默认 kernel/host 文件、CMake 项目、测试源码、结果路径和打包清单。统一脚本从该文件读取路径和默认选择；命令行只需传入配置文件，必要时才覆盖后端或文件选择。

## 工具要求

- Python 3.9+ 和 PyYAML：`python3 -m pip install -r requirements.txt`
- CMake 3.21+
- CUDA：兼容的 CUDA Toolkit/nvcc、C++17 host 编译器，以及设备测试所需驱动/设备
- CANN：匹配的 CANN toolkit/bisheng、ACL runtime 与 Ascend NPU；add CANN v1 已在 Atlas 200I DK A2 / Ascend 310B4、CANN 7.0 RC1 上验证

## 查看配置和源文件

从 WSL/Linux 仓库根目录运行：

```bash
python3 tools/ops.py list operators/add/operator.yaml
```

`operator.yaml` 当前默认后端为 CUDA，默认选择 `add_kernel_v2.cu` 和 `add_host_v2.cu`。默认文件可在该配置中修改。也可以临时通过 `--backend`、`--kernel`、`--host` 覆盖；文件名由开发者决定，不要求固定命名格式。

## 构建与测试

```bash
python3 tools/ops.py build operators/add/operator.yaml --examples
python3 tools/ops.py test operators/add/operator.yaml
```

切换到 v1 文件组合：

```bash
python3 tools/ops.py build operators/add/operator.yaml \
  --kernel add_kernel_v1.cu --host add_host_v1.cu --examples
python3 tools/ops.py test operators/add/operator.yaml \
  --kernel add_kernel_v1.cu --host add_host_v1.cu
```

直接使用 CMake 时，配置文件中的文件路径对应如下参数：

```bash
cmake -S operators/add -B build/add/cuda/manual \
  -DOPS_ENABLE_CUDA=ON -DOPS_ENABLE_CANN=OFF \
  -DOPS_KERNEL_FILE=add_kernel_v2.cu -DOPS_HOST_FILE=add_host_v2.cu \
  -DOPS_BUILD_EXAMPLES=ON
cmake --build build/add/cuda/manual --config Release
```

测试工程由 `operator.yaml` 中 `backends.cuda.tests.project` 指定，正确性和显存评估由该工程的 CTest suites 执行。没有 `nvcc` 时配置明确失败；没有设备或驱动时设备用例明确 Skip。

## 性能评估

正确性通过后运行配置中指定的 benchmark：

```bash
python3 tools/ops.py benchmark operators/add/operator.yaml
```

输入规模和迭代次数从 `operator.yaml` 的 `backends.cuda.benchmark.sizes` 与 `iterations` 读取。修改配置即可调整常用测量范围；`--sizes` 和 `--iterations` 仍可用于单次临时覆盖。

结果追加到 `backends.cuda.tests.results.performance` 指定的目录，文件名根据 kernel/host 文件名生成。显存评估写入配置指定的 `memory` 路径。记录包含具体源码、设备、输入规模、编译器/runtime、资源读数和测量数据。

## CANN 与打包

CANN 可通过配置文件切换后端：

```bash
python3 tools/ops.py build operators/add/operator.yaml --backend cann
```

板端先加载 CANN 环境并执行构建、测试、benchmark：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
python3 tools/ops.py build operators/add/operator.yaml --backend cann --examples
python3 tools/ops.py test operators/add/operator.yaml --backend cann
python3 tools/ops.py benchmark operators/add/operator.yaml --backend cann
```

CANN v1 目前针对已验证的 dav-m300，使用 bisheng + ACL Runtime，不依赖 PyTorch。其他 toolkit/SoC 未验证时不能据此标记可用。

打包使用同一配置和当前默认源码选择：

```bash
python3 tools/ops.py package operators/add/operator.yaml
```

输出位于 `dist/`。源码包不等于已编译二进制包。旧的 `tests/cuda/performance/results/cuda.csv` 是历史格式，工具不会覆盖它。
