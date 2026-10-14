# add 使用说明

## 配置与源码选择

operator.yaml 是构建、测试、benchmark 和打包配置入口，包含 CUDA/CANN 源码目录、默认 kernel/host 文件、测试工程、benchmark 尺寸和结果路径。文件名只表示源码选择值，可继续增加 add_kernel_v3.cpp 等版本：

    python3 tools/ops.py list operators/add/operator.yaml

## CUDA

在安装 CUDA Toolkit、nvcc、CMake 3.21+ 的主机执行：

    python3 tools/ops.py build operators/add/operator.yaml --backend cuda --examples
    python3 tools/ops.py test operators/add/operator.yaml --backend cuda
    python3 tools/ops.py benchmark operators/add/operator.yaml --backend cuda
    ./build/add/cuda/add_kernel_v2__add_host_v2/add_demo

要选择另一组 kernel/host，可使用 --kernel 和 --host 临时覆盖，或改 operator.yaml 中对应的 default_file。

## CANN

CANN 已在 Atlas 200I DK A2 / Ascend 310B4、CANN 7.0 RC1 上实测。板端先加载 toolkit 环境，然后执行：

    source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
    python3 tools/ops.py build operators/add/operator.yaml --backend cann --examples
    python3 tools/ops.py test operators/add/operator.yaml --backend cann
    python3 tools/ops.py benchmark operators/add/operator.yaml --backend cann
    ./build/add/cann/add_kernel_v1__add_host_v1/add_demo

CANN build 使用配置选择的 kernel/host 源文件，以 bisheng -x cce 构建共享库；当前 arch 设置为已验证板卡的 dav-m300。host API 接收 ACL device pointers 和 stream，负责入队，不负责分配或同步；调用方负责内存生命周期和同步。测试使用 ACL Runtime，不依赖 PyTorch。

当前 CANN v1 为 CANN 7.0 RC1 目标可编译的标量 CCE 示例。它尚未在其他 Ascend SoC 或 SDK 上验证。缺少 CANN compiler/toolkit 时 CMake 给出明确错误；缺少运行设备时 CTest 对设备用例标为 Skip。

## 测试、内存与性能

CUDA/CANN 测试均在各自的 tests/<backend>/ 工程中。test 运行正确性和显存/设备内存评估；benchmark 根据 YAML 中 sizes、iterations 配置运行。CANN 当前配置为 [1024, 65536, 1048576] 个 float32 元素、每个规模 30 次 kernel 调用。事件计时后同步，延迟为平均设备执行时间；有效带宽用 12*N / latency 估算，不是内存控制器硬件计数器数据。

CSV 结果在 tests/<backend>/<memory|performance>/results/。只有实际运行的测量会写入，设备、编译器/runtime、尺寸、显存、延迟和校验状态均随记录保留。

## 打包

    python3 tools/ops.py package operators/add/operator.yaml --backend cann

源码交付整理在 dist/。源码包本身不代表已经针对目标设备编译或验证。
