# add 使用说明

## 配置入口

`operator.yaml` 包含算子根目录、后端 kernel/host 目录、默认源码文件、CMake 项目、tests 源码和结果路径，以及打包文件清单。当前默认后端是 CUDA，默认 kernel/host 为 `add_kernel_v2.cu` 和 `add_host_v2.cu`。这些文件名只是当前选择值；你可以在配置中改默认项，或在命令行临时覆盖。

查看配置里各后端的源码文件：

```bash
python3 tools/ops.py list operators/add/operator.yaml
```

## 构建、测试和 demo

安装工具脚本依赖后，从仓库根目录构建默认配置、运行测试并启动 demo：

```bash
python3 -m pip install -r requirements.txt
python3 tools/ops.py build operators/add/operator.yaml --examples
python3 tools/ops.py test operators/add/operator.yaml
./build/add/cuda/add_kernel_v2__add_host_v2/add_demo
```

若要临时选择 v1 文件组合，不必修改配置：

```bash
python3 tools/ops.py build operators/add/operator.yaml \
  --kernel add_kernel_v1.cu --host add_host_v1.cu --examples
python3 tools/ops.py test operators/add/operator.yaml \
  --kernel add_kernel_v1.cu --host add_host_v1.cu
./build/add/cuda/add_kernel_v1__add_host_v1/add_demo
```

host 与 kernel 需要按实现接口配对；CMake 会把配置或命令选择的 kernel 文件包含到所选 host 编译单元中。直接使用 CMake 时，传入相同文件名：

```bash
cmake -S operators/add -B build/add/cuda/manual \
  -DOPS_ENABLE_CUDA=ON -DOPS_ENABLE_CANN=OFF \
  -DOPS_KERNEL_FILE=add_kernel_v2.cu -DOPS_HOST_FILE=add_host_v2.cu \
  -DOPS_BUILD_EXAMPLES=ON
cmake --build build/add/cuda/manual --config Release
```

正确性测试覆盖 `N=0`、`N=1`、`N=259`、非法参数和未对齐输入。显存 suite 报告三缓冲申请量和设备内存读数，写入配置中的 `backends.cuda.tests.results.memory`。没有设备/驱动时设备用例显示 `SKIP`，不计为通过。构建目录按源码选择隔离。

## 性能评估

正确性通过后运行配置中的 benchmark：

```bash
python3 tools/ops.py benchmark operators/add/operator.yaml \
  --sizes 1024,1048576,16777216 --iterations 100
```

这会评估配置当前选择的源码组合。比较另一组实现时覆盖 kernel 和 host：

```bash
python3 tools/ops.py benchmark operators/add/operator.yaml \
  --kernel add_kernel_v1.cu --host add_host_v1.cu \
  --sizes 1024,1048576,16777216 --iterations 100
```

默认 CSV 写入 `tests/cuda/performance/results/<kernel-stem>__<host-stem>.csv`，记录所选文件、设备、输入规模、编译器/runtime/driver、显存读数、延迟、估算有效带宽和校验状态。float32 加法按每元素读 A、读 B、写 C 估算 `12 * N` 字节；有效带宽不是硬件计数器测得的 DRAM 带宽。工具不会覆盖旧格式 `cuda.csv`。

## CANN 状态

`src/cann/` 和 `tests/cann/` 当前为 scaffold。`operator.yaml` 已留有 kernel/host/CMake/tests 路径配置，但没有实现源文件；脚本会明确提示缺少源码，不会声称可构建或可运行。

## 打包

```bash
python3 tools/ops.py package operators/add/operator.yaml
```

包会按当前选择写入 `dist/`，包含配置列出的构建、测试、文档文件和 backend 源码。源码包不表示当前机器已构建或通过设备验证。
