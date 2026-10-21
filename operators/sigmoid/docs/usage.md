# sigmoid 使用说明

## 源码选择与状态

`operator.yaml` 是构建、测试、benchmark 和打包配置入口。CUDA 默认选择 `*_kernel_v2.cu` / `*_host_v2.cu`，CANN 默认选择 v1；可通过 `--kernel` 与 `--host` 选择其他源码组合。文件名只是用户管理源码的命名习惯。

本算子当前两个后端均为 scaffold，尚未在对应设备验证。后端工具会先检查本机是否有对应编译器：CUDA 需 nvcc；CANN 需 bisheng 并加载 toolkit 环境。

## CUDA

在具备 CUDA Toolkit、nvcc、CMake 3.21+ 和 GPU 的主机运行：

```sh
python3 tools/operator_cuda.py list operators/sigmoid/operator.yaml
python3 tools/operator_cuda.py build operators/sigmoid/operator.yaml --examples
python3 tools/operator_cuda.py test operators/sigmoid/operator.yaml
python3 tools/operator_cuda.py benchmark operators/sigmoid/operator.yaml
./build/sigmoid/cuda/sigmoid_kernel_v2__sigmoid_host_v2/sigmoid_demo
```

## CANN

在目标 Ascend 主机加载 toolkit 环境后运行：

```sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 tools/operator_cann.py list operators/sigmoid/operator.yaml
python3 tools/operator_cann.py build operators/sigmoid/operator.yaml --examples
python3 tools/operator_cann.py test operators/sigmoid/operator.yaml
python3 tools/operator_cann.py benchmark operators/sigmoid/operator.yaml
./build/sigmoid/cann/sigmoid_kernel_v1__sigmoid_host_v1/sigmoid_demo
```

v1 每轮处理一个元素；v2 展开四个标量元素并检查尾部。CANN kernel 使用本仓库现有低层 CCE/host 结构。目标架构设置和源码存在不表示已经编译或验证；若编译器不支持所用数学函数，应在目标 CANN 版本上据实际诊断调整。

## 正确性、内存和性能

`test` 构建并运行正确性及设备内存评估；无设备时用例应显示 Skip，不计作通过。`benchmark` 的规模、迭代数从 YAML 读取，结果写入 `tests/<backend>/performance/results/`；仅真实运行会产生记录。工作集为 `2*N*sizeof(float)`，估算有效带宽按输入读取加输出写入计算，不是硬件计数器实测。

## 打包

```sh
python3 tools/operator_cuda.py package operators/sigmoid/operator.yaml
python3 tools/operator_cann.py package operators/sigmoid/operator.yaml
```

交付目录位于 `dist/`，包内容由 YAML 的 `package` 清单控制。
