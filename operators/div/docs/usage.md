# div 用法

所有运行统一使用仓库根目录的 `tools/run_operator.py`。该算子的后端状态、默认实现文件、bench 参数与 profile 参数以 `operator.yaml` 为准。

## test

逐一验证四种实现；依赖、编译器或设备缺失时会清楚失败或报告 Skip，不会使用 CPU 回退：

    python3 tools/run_operator.py test operators/div/operator.yaml --backend pytorch
    python3 tools/run_operator.py test operators/div/operator.yaml --backend triton
    python3 tools/run_operator.py test operators/div/operator.yaml --backend cuda
    python3 tools/run_operator.py test operators/div/operator.yaml --backend cann

CUDA/CANN 可通过 `--kernel <文件>` 与 `--host <文件>` 选择兼容源码组合。测试覆盖正确性和设备内存分配检查；测试数据和实现约束见 `spec.md`。

## bench

    python3 tools/run_operator.py bench operators/div/operator.yaml --backend pytorch
    python3 tools/run_operator.py bench operators/div/operator.yaml --backend triton
    python3 tools/run_operator.py bench operators/div/operator.yaml --backend cuda
    python3 tools/run_operator.py bench operators/div/operator.yaml --backend cann

规模和迭代数默认从 YAML 读取，也可仅覆盖本次运行，例如：

    python3 tools/run_operator.py bench operators/div/operator.yaml --backend cuda --kernel <kernel-file> --host <host-file> --sizes 1024,1048576 --iterations 100

结果只在真实运行后追加到 `tests/<backend>/bench/results/`。记录包含设备、规模、dtype、所选实现、工具链/runtime、延迟和带宽估算。估算带宽不等于硬件计数器读取值。

## profile

    python3 tools/run_operator.py profile operators/div/operator.yaml --backend pytorch
    python3 tools/run_operator.py profile operators/div/operator.yaml --backend triton
    python3 tools/run_operator.py profile operators/div/operator.yaml --backend cuda
    python3 tools/run_operator.py profile operators/div/operator.yaml --backend cann

PyTorch/Triton 使用 `torch.profiler`；CUDA 使用 Nsight Compute CLI（`ncu`）；CANN 使用板端 `msprof`。profile 参数来自 YAML，也可用 `--size` 与 `--iterations` 覆盖。trace/report 写入被忽略的 `build/${op}/profile/<backend>/`，不作为源码提交。Profile 时间不替代 bench 结果。

## 构建与打包

CUDA/CANN 的独立 build 和 package 入口保留：

    python3 tools/operator_cuda.py build operators/div/operator.yaml
    python3 tools/operator_cann.py build operators/div/operator.yaml
    python3 tools/operator_cuda.py package operators/div/operator.yaml
    python3 tools/operator_cann.py package operators/div/operator.yaml

PyTorch/Triton 需要 CUDA-enabled PyTorch；Triton 版本还需安装 Triton。CUDA 需要 nvcc 和 Nsight Compute 才能 profile；CANN 命令需先加载目标板匹配的 CANN 环境。后端当前验证状态以 operator.yaml 为准；CANN scaffold 后端因 dav-m300 API/SDK 不匹配而不能执行。已验证状态不表示 Nsight Compute 硬件计数器采集成功（本轮受 WSL 计数器权限限制）。
