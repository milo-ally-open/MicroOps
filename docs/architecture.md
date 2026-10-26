# 架构概览

仓库按算子组织源码、测试工程和文档。一个算子目录包含 CUDA/CANN kernel 与 host、PyTorch/Triton GPU 实现、示例、动画及统一测试目录：

    operators/<op>/
    ├── src/{pytorch,triton,cuda,cann}/
    ├── tests/{pytorch,triton,cuda,cann}/{test,bench,profile}/
    ├── docs/
    ├── examples/
    └── operator.yaml

`operator.yaml` 是工具入口的元数据来源：实现源码、默认选择、test/bench工程、bench规模/迭代数、profile采集参数、结果路径和package清单都在此登记。完整语义只在 `docs/spec.md` 维护，不在 YAML 重复规格。

## 统一执行入口

`tools/run_operator.py` 面向 PyTorch、Triton、CUDA、CANN 四类 GPU/NPU 实现统一提供 `test`、`bench`、`profile`：

- test 验证正确性、dtype/shape/API约束，并在原生 CUDA/CANN 测试工程中检查设备分配。
- bench 采集实际运行延迟、设备和版本信息、工作集及带宽估算。
- profile 调用 torch.profiler、Nsight Compute 或 CANN msprof，报告写入忽略的 build/。

CUDA 与 CANN 的 CMake 编译/打包仍由 `tools/operator_cuda.py` 与 `tools/operator_cann.py` 负责；两者各自加载配置，默认构建不强制另一套工具链。测试工程只编译选中的后端。

`test`、`bench` 结果保存在算子 tests 对应后端的 stage 目录。profile traces/reports 是生成物，保存在 `build/<op>/profile/<backend>/`，不放入源码目录。后端只有完成实际设备验证后才登记为 device-tested；仅有源码、配置或编译器不表示已验证。
