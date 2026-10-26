# 构建、test、bench 与 profile

仓库根目录下用同一个入口运行四种 GPU/NPU 后端：

    python3 tools/run_operator.py <test|bench|profile> operators/<op>/operator.yaml --backend <pytorch|triton|cuda|cann>

源文件默认选择来自 `operator.yaml`。CUDA/CANN 可用 `--kernel` 和 `--host` 指定特定版本；PyTorch/Triton 的源码入口同样登记在 YAML。

## 工具链

- PyTorch：CUDA-enabled PyTorch、可用 NVIDIA GPU。
- Triton：PyTorch CUDA、Triton 与可用 NVIDIA GPU。
- CUDA：CUDA Toolkit/nvcc、C++17 编译器、驱动与 GPU；profile 另需 Nsight Compute CLI（ncu）。
- CANN：在目标板加载匹配的 CANN toolkit 环境，需 bisheng、ACL runtime 和 Ascend NPU；profile 另需 msprof。

普通 CPU 环境可查阅配置和文档，但不能运行这些 GPU/NPU 实现。缺少依赖时命令会明确退出，不回退到 CPU。

## CUDA

构建算子库/示例仍使用 CUDA 专用构建入口；test、bench、profile 使用统一入口：

    python3 tools/operator_cuda.py build operators/add/operator.yaml --examples
    python3 tools/run_operator.py test operators/add/operator.yaml --backend cuda
    python3 tools/run_operator.py bench operators/add/operator.yaml --backend cuda
    python3 tools/run_operator.py profile operators/add/operator.yaml --backend cuda

选 kernel/host 版本：

    python3 tools/operator_cuda.py build operators/add/operator.yaml --kernel add_kernel_v1.cu --host add_host_v1.cu
    python3 tools/run_operator.py test operators/add/operator.yaml --backend cuda --kernel add_kernel_v1.cu --host add_host_v1.cu
    python3 tools/run_operator.py bench operators/add/operator.yaml --backend cuda --kernel add_kernel_v1.cu --host add_host_v1.cu

## CANN

在开发板或 CANN 环境加载 Toolkit 后运行：

    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python3 tools/operator_cann.py build operators/add/operator.yaml --examples
    python3 tools/run_operator.py test operators/add/operator.yaml --backend cann
    python3 tools/run_operator.py bench operators/add/operator.yaml --backend cann
    python3 tools/run_operator.py profile operators/add/operator.yaml --backend cann

示例板端 CANN 环境和版本须以实际 `set_env.sh` 为准。CANN profile 通过 msprof 包装测试工程构建出的 bench executable，报告进入 `build/add/profile/cann/`。

## PyTorch 与 Triton

    python3 tools/run_operator.py test operators/add/operator.yaml --backend pytorch
    python3 tools/run_operator.py bench operators/add/operator.yaml --backend pytorch
    python3 tools/run_operator.py profile operators/add/operator.yaml --backend pytorch
    python3 tools/run_operator.py test operators/add/operator.yaml --backend triton
    python3 tools/run_operator.py bench operators/add/operator.yaml --backend triton
    python3 tools/run_operator.py profile operators/add/operator.yaml --backend triton

PyTorch 与 Triton 的九个算子已在 RTX 4060 Laptop GPU 上执行 test、bench 与 `torch.profiler` profile。profile 输出 Chrome trace、摘要和 metadata。测量受同时运行的 GPU 工作负载影响，比较结果应结合多次同条件测量。

## 配置、结果和打包

`operator.yaml` 的 `backends.<backend>.bench` 与 `frameworks.<backend>.bench` 保存默认规模、迭代和结果目录；profile 节保存采集参数。命令行参数只覆盖单次运行，不改配置。

- test 结果与设备分配记录：`operators/<op>/tests/<backend>/test/results/`
- bench CSV：`operators/<op>/tests/<backend>/bench/results/`
- profile trace/report：`build/<op>/profile/<backend>/`（忽略生成目录）
- CUDA/CANN 源码包：分别用 `tools/operator_cuda.py package` 和 `tools/operator_cann.py package` 输出到 `dist/`

正确性检查、bench 测量和 profile 采集是三个独立结论；不要用 profiler 期间的时间填 bench 结果。实测数据按运行时设备、编译器/runtime、输入规模和源码选择记录，未执行的数据留空。
