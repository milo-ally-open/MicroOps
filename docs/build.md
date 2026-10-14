# 构建与验证

## 要求

- CMake 3.21+
- C++17 编译器
- Python 3.9+（使用仓库统一脚本时）
- 可选 CUDA/Ascend C 依赖仅在显式启用相应后端时需要

## CPU-only

```powershell
python tools/build.py --backend cpu
python tools/test.py --backend cpu
```

默认 CMake 配置只生成 `reduce_sum_reference` 和 CPU 测试。直接使用 CMake 时：

```powershell
cmake -S . -B build/cpu -DOPS_ENABLE_CUDA=OFF -DOPS_ENABLE_ASCENDC=OFF
cmake --build build/cpu
ctest --test-dir build/cpu --output-on-failure
```

## 独立后端构建

```powershell
python tools/build.py --backend cuda
python tools/build.py --backend ascendc
```

等价 preset 为 `cuda` 和 `ascendc`。这两个 preset 使用独立 build 目录。Ascend C 当前只有 scaffold，构建入口会停止并给出原因。CUDA 构建要求 CMake 能找到 CUDA 编译器和 Toolkit；当前 CUDA 源码未在本机验证。不要通过关闭依赖检查来声称构建成功。

## 测试、benchmark 和包

`python tools/test.py` 会构建并运行 CTest。请求设备后端时，脚本分别报告 CPU 测试结果与未实现/不可运行的设备测试状态。`python tools/benchmark.py --backend cpu` 会实际运行 CPU benchmark，并将设备、编译器、运行时、输入规模和时间写入 CSV；没有运行的数据不生成测量行。`python tools/package.py --operator reduce_sum --backend cpu` 将所选实现、元数据和使用说明复制到 `dist/`。

本机初始化时没有 CMake、Python、CUDA Toolkit、CANN/Ascend C 编译器或可确认的设备，因此无法报告编译、测试或性能结果。安装依赖后可按以上命令验证。
