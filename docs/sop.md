# 算子开发 SOP

本文是新增或改造算子的统一工作流，适用于逐元素算子、归约、卷积和 Attention。每个算子按同一顺序完成定义、实现、测试、性能测量、profiling、文档和打包。当前每个算子可包含 PyTorch、Triton、CUDA、CANN 四种 GPU/NPU 实现；不提供 CPU 执行回退。

## 1. 检查工作区和目标环境

先确认当前改动与工具链状态，保留工作区中已有文件：

    git status --short --branch
    python3 --version
    python3 -m pip install -r requirements.txt

按本次目标确认对应设备与工具链：PyTorch/Triton 需要 CUDA-enabled PyTorch 和 NVIDIA GPU；CUDA 需要 nvcc 与 CUDA GPU；CANN 需要加载匹配的 CANN 环境、bisheng 与 Ascend 设备。没有安装的工具链和设备视为不可用，不把源码存在或配置成功称作后端通过。

## 2. 定义算子与实现版本

在 `operators/<op>/docs/` 定义数学语义、输入输出、layout、dtype、shape、边界、数值精度、别名规则、后端和验收标准。每个可选实现由 `operator.yaml` 登记；CUDA/CANN 可选择 kernel 与 host 文件，PyTorch/Triton 可选择对应入口。v1/v2 等名称只是源码版本名，不代表自动通过或一定更快。

以 add v1 为例：连续 float32 向量满足 `C[i] = A[i] + B[i]`；shape 为 `[N]`，不支持广播或 stride。test 覆盖空输入、单元素、尾部元素和错误参数。该语义不限定线程块、每线程元素数或 launch 策略。

## 3. 目录和交付件

算子实现位于 `operators/<op>/src/{pytorch,triton,cuda,cann}/`，测试项目位于：

    operators/<op>/tests/<backend>/{test,bench,profile}/

其中：

- `test/` 保存正确性和设备分配检查源码。
- `bench/` 保存性能测量源码和真实 CSV。
- `profile/` 保留目录占位；profile 生成的 trace/report 放在仓库忽略的 `build/<op>/profile/<backend>/`。
- 算子定义和用法文档仅放在该算子 `docs/`，小 demo 放在 `examples/`。

CUDA/CANN 使用后端 CMake 测试工程；PyTorch/Triton 通过 GPU tensor API 执行。输出数据只在实际执行后记录。

## 4. 统一 test、bench、profile 入口

所有测试、性能测量和 profiling 统一使用：

    python3 tools/run_operator.py <test|bench|profile> operators/<op>/operator.yaml --backend <pytorch|triton|cuda|cann>

### test：验证语义与设备分配

    python3 tools/run_operator.py test operators/add/operator.yaml --backend cuda
    python3 tools/run_operator.py test operators/add/operator.yaml --backend cann
    python3 tools/run_operator.py test operators/add/operator.yaml --backend pytorch
    python3 tools/run_operator.py test operators/add/operator.yaml --backend triton

CUDA/CANN test 会配置、构建并运行算子正确性与设备内存检查；PyTorch/Triton test 检查 GPU-only 输入输出契约、shape、dtype、边界样本与精度。缺少依赖或设备时命令明确失败或报告 Skip，不将 Skip 计为通过。

### bench：采集可比较的真实测量

    python3 tools/run_operator.py bench operators/add/operator.yaml --backend cuda
    python3 tools/run_operator.py bench operators/add/operator.yaml --backend cann
    python3 tools/run_operator.py bench operators/add/operator.yaml --backend pytorch
    python3 tools/run_operator.py bench operators/add/operator.yaml --backend triton

默认输入规模、预热和迭代次数从 `operator.yaml` 读取；可临时覆盖，例如：

    python3 tools/run_operator.py bench operators/add/operator.yaml --backend cuda --kernel add_kernel_v1.cu --host add_host_v1.cu --sizes 1024,1048576 --iterations 100

bench CSV 位于 `tests/<backend>/bench/results/`，需记录设备、输入规模、dtype、所选实现、工具链/runtime 版本、延迟、分配状态及有效带宽估算。带宽计算口径写入文档；估算带宽不等同于硬件计数器 DRAM 带宽。比较优化版本时应保持设备、输入、编译选项和计时边界一致。

### profile：分析 kernel 和运行时活动

    python3 tools/run_operator.py profile operators/add/operator.yaml --backend cuda
    python3 tools/run_operator.py profile operators/add/operator.yaml --backend cann
    python3 tools/run_operator.py profile operators/add/operator.yaml --backend pytorch
    python3 tools/run_operator.py profile operators/add/operator.yaml --backend triton

CUDA 使用 Nsight Compute CLI `ncu`；CANN 在板端使用 `msprof`；PyTorch/Triton 使用 `torch.profiler` 导出 trace、统计摘要和元数据。profile 结果写入 `build/<op>/profile/<backend>/`，不提交为源码。profiling 会扰动执行，不用 profile 时间替代 bench 数据。工具缺失、设备不可用或采集失败时命令报错，不伪造报告。

## 5. add v1 示例和打包

add v1 的最小 CUDA kernel 是每线程处理一个元素。实现后依次运行 test、bench，再按需要 profile；三者回答不同问题：正确性是否满足、在规定输入下执行多快、时间/资源消耗集中在哪里。CUDA/CANN 的普通 build 和 package 仍由对应工具负责：

    python3 tools/operator_cuda.py build operators/add/operator.yaml
    python3 tools/operator_cann.py build operators/add/operator.yaml
    python3 tools/operator_cuda.py package operators/add/operator.yaml
    python3 tools/operator_cann.py package operators/add/operator.yaml

包写到 `dist/`，构建和 profile 文件写到 `build/`。提交前核对 YAML 路径、spec、test/bench/profile 命令与结果，运行可用的 test 和 `git diff --check`。报告 PASS、FAIL、SKIP 及未验证的后端；后端状态只有在目标环境真实验证后才能更新。
