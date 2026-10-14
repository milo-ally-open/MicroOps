# reduce_sum 使用说明

## CPU 参考实现

头文件为 `reference/include/ops/reduce_sum.hpp`，实现位于 `reference/src/reduce_sum.cpp`。接口接受 `const float*` 和 `size_t`，返回 float。空输入返回零；非空空指针抛出 `std::invalid_argument`。

在仓库根目录执行：

```powershell
python tools/build.py --backend cpu
python tools/test.py --backend cpu
```

直接使用 CMake 时默认只配置 CPU：

```powershell
cmake -S . -B build/cpu -DOPS_ENABLE_CUDA=OFF -DOPS_ENABLE_ASCENDC=OFF
cmake --build build/cpu
ctest --test-dir build/cpu --output-on-failure
```

## CUDA

CUDA 候选 kernel 在 `backends/cuda/kernel/reduce_sum_kernel.cuh`，host wrapper 在 `backends/cuda/host/reduce_sum_cuda.cu`。在 CUDA Toolkit 与设备环境中可用 `python tools/build.py --backend cuda` 单独配置。当前尚无设备运行测试，源码状态仍为 scaffold_unverified；构建或存在编译器都不等于设备验证完成。

## Ascend C

`backends/ascendc/` 目前只有标明未实现的入口。`python tools/build.py --backend ascendc` 会明确停止并报告原因；不会创建一个空成功目标。待选定 CANN/Ascend C API 并实现 host/kernel 后，再启用实际构建。

## Demo

`examples/grayscale_mean.cpp` 将 RGB 像素转换为灰度值，然后把灰度数组传给 `reduce_sum`，最后除以像素数得到均值。该 demo 使用 CPU 参考实现，不依赖图像 SDK。

```powershell
python tools/build.py --backend cpu --examples
.
build\cpu\reduce_sum_grayscale_mean.exe
```

Visual Studio 等多配置生成器会将 Release demo 放在 `build/cpu/Release/`；单配置生成器直接放在 `build/cpu/`。

## 测试、benchmark 与打包

```powershell
python tools/test.py --backend cpu
python tools/benchmark.py --backend cpu --sizes 1024,1048576
python tools/package.py --operator reduce_sum --backend cpu
```

benchmark 将真实测量写至 `build/benchmark_results.csv`，包含设备、编译器/运行时和输入规模。包位于 `dist/reduce_sum-0.1.0-cpu/`；存在同名输出时需显式传 `--force` 才会替换。
