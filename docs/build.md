# 构建、测试与评估

每个算子的 `operator.yaml` 记录后端源码目录、默认 kernel/host 文件、CMake 项目、测试源码、结果路径和打包清单。CUDA 与 CANN 使用独立脚本，命令不再需要 `--backend`；源码选择可从 YAML 读取，也可通过 `--kernel`、`--host` 临时覆盖。

## 工具要求

- Python 3.9+ 和 PyYAML：`python3 -m pip install -r requirements.txt`
- CMake 3.21+
- CUDA：兼容的 CUDA Toolkit/nvcc、C++17 host 编译器，以及设备测试所需驱动/设备
- CANN：匹配的 CANN toolkit/bisheng、ACL runtime 与 Ascend NPU；add CANN v1/v2 已在 Atlas 200I DK A2 / Ascend 310B4、CANN 7.0 RC1 上验证

## 查看源码

从 WSL/Linux 仓库根目录运行对应脚本：

```bash
python3 tools/operator_cuda.py list operators/add/operator.yaml
python3 tools/operator_cann.py list operators/add/operator.yaml
```

默认文件和候选扩展名来自 `operator.yaml`。文件名由开发者决定，不要求固定命名格式；目录内有多个候选文件时，通过 `--kernel` 和 `--host` 选择组合。

## CUDA 构建与测试

```bash
python3 tools/operator_cuda.py build operators/add/operator.yaml --examples
python3 tools/operator_cuda.py test operators/add/operator.yaml
```

选择 v1 文件组合：

```bash
python3 tools/operator_cuda.py build operators/add/operator.yaml \
  --kernel add_kernel_v1.cu --host add_host_v1.cu --examples
python3 tools/operator_cuda.py test operators/add/operator.yaml \
  --kernel add_kernel_v1.cu --host add_host_v1.cu
```

直接使用 CMake 时，文件路径对应如下参数：

```bash
cmake -S operators/add -B build/add/cuda/manual \
  -DOPS_ENABLE_CUDA=ON -DOPS_ENABLE_CANN=OFF \
  -DOPS_KERNEL_FILE=add_kernel_v2.cu -DOPS_HOST_FILE=add_host_v2.cu \
  -DOPS_BUILD_EXAMPLES=ON
cmake --build build/add/cuda/manual --config Release
```

测试工程由 `operator.yaml` 中 `backends.cuda.tests.project` 指定。没有 `nvcc` 时配置明确失败；没有设备或驱动时设备用例明确 Skip。

## CUDA 性能评估

正确性通过后运行配置中的 benchmark：

```bash
python3 tools/operator_cuda.py benchmark operators/add/operator.yaml
```

输入规模和迭代次数从 `backends.cuda.benchmark.sizes` 与 `iterations` 读取；可修改 YAML 设置常用范围，也可使用 `--sizes`、`--iterations` 临时覆盖。结果追加到配置指定的目录，文件名由 kernel/host 文件组合生成，记录设备、输入规模、编译器/runtime、资源读数和测量数据。

## CANN 构建、测试与性能评估

在板端加载 CANN 环境后执行：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
python3 tools/operator_cann.py build operators/add/operator.yaml --examples
python3 tools/operator_cann.py test operators/add/operator.yaml
python3 tools/operator_cann.py benchmark operators/add/operator.yaml
```

选择 CANN v2 文件组合：

```bash
python3 tools/operator_cann.py build operators/add/operator.yaml \
  --kernel add_kernel_v2.cpp --host add_host_v2.cpp --examples
python3 tools/operator_cann.py test operators/add/operator.yaml \
  --kernel add_kernel_v2.cpp --host add_host_v2.cpp
python3 tools/operator_cann.py benchmark operators/add/operator.yaml \
  --kernel add_kernel_v2.cpp --host add_host_v2.cpp
```

CANN 当前针对已验证的 dav-m300，使用 bisheng + ACL Runtime，不依赖 PyTorch。其他 toolkit/SoC 未验证时不能据此标记可用。

## 打包

每个后端分别打包：

```bash
python3 tools/operator_cuda.py package operators/add/operator.yaml
python3 tools/operator_cann.py package operators/add/operator.yaml
```

输出位于 `dist/`。源码包不等于已编译二进制包；工具不会覆盖历史测试结果文件。
