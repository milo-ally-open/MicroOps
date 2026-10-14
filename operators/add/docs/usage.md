# add 使用说明

## 配置与源码选择

operator.yaml 是构建、测试、benchmark 和打包配置入口，包含 CUDA/CANN 源码目录、默认 kernel/host 文件、测试工程、benchmark 尺寸和结果路径。文件名只表示源码选择值，可继续增加 add_kernel_v3.cpp 等版本：

CUDA 与 CANN 脚本会先检查各自编译器（`nvcc` 或 `bisheng`）是否在 PATH 中；缺少对应环境时，包括 list 和 package 在内的命令都会明确退出。

    python3 tools/operator_cuda.py list operators/add/operator.yaml
    python3 tools/operator_cann.py list operators/add/operator.yaml

## CUDA

在安装 CUDA Toolkit、nvcc、CMake 3.21+ 的主机执行：

    python3 tools/operator_cuda.py build operators/add/operator.yaml --examples
    python3 tools/operator_cuda.py test operators/add/operator.yaml
    python3 tools/operator_cuda.py benchmark operators/add/operator.yaml
    ./build/add/cuda/add_kernel_v2__add_host_v2/add_demo

要选择另一组 kernel/host，可使用 --kernel 和 --host 临时覆盖，或改 operator.yaml 中对应的 default_file。

## CANN

CANN 已在 Atlas 200I DK A2 / Ascend 310B4、CANN 7.0 RC1 上实测。板端先加载 toolkit 环境，然后执行：

    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python3 tools/operator_cann.py build operators/add/operator.yaml --examples
    python3 tools/operator_cann.py test operators/add/operator.yaml
    python3 tools/operator_cann.py benchmark operators/add/operator.yaml
    ./build/add/cann/add_kernel_v1__add_host_v1/add_demo

CANN build 使用配置选择的 kernel/host 源文件，以 bisheng -x cce 构建共享库；当前 arch 设置为已验证板卡的 dav-m300。host API 接收 ACL device pointers 和 stream，负责入队，不负责分配或同步；调用方负责内存生命周期和同步。测试使用 ACL Runtime，不依赖 PyTorch。

CANN v1/v2 都是 CANN 7.0 RC1 与 dav-m300 上验证的 CCE 示例：v1 逐元素计算，v2 每轮展开四个元素并检查尾部。选择 v2 可使用：

    python3 tools/operator_cann.py build operators/add/operator.yaml --kernel add_kernel_v2.cpp --host add_host_v2.cpp --examples
    python3 tools/operator_cann.py test operators/add/operator.yaml --kernel add_kernel_v2.cpp --host add_host_v2.cpp
    python3 tools/operator_cann.py benchmark operators/add/operator.yaml --kernel add_kernel_v2.cpp --host add_host_v2.cpp
    ./build/add/cann/add_kernel_v2__add_host_v2/add_demo

版本号只用于区分可选源码；v2 的多元素展开不等同于硬件 SIMD。两版均未在其他 Ascend SoC 或 SDK 上验证。缺少 CANN compiler/toolkit 时 CMake 给出明确错误；缺少运行设备时 CTest 对设备用例标为 Skip。

## 测试、内存与性能

CUDA/CANN 测试均在各自的 tests/<backend>/ 工程中。test 运行正确性和显存/设备内存评估；benchmark 根据 YAML 中 sizes、iterations 配置运行。CANN benchmark 配置为 [1024, 65536, 1048576] 个 float32 元素、每个规模 30 次 kernel 调用；v1/v2 分别生成以所选 kernel/host 组合命名的 CSV。事件计时后同步，延迟为平均设备执行时间；有效带宽用 12*N / latency 估算，不是内存控制器硬件计数器数据。

CSV 结果在 tests/<backend>/<memory|performance>/results/。只有实际运行的测量会写入，设备、编译器/runtime、尺寸、显存、延迟和校验状态均随记录保留。

## 打包

    python3 tools/operator_cann.py package operators/add/operator.yaml

源码交付整理在 dist/。源码包本身不代表已经针对目标设备编译或验证。
