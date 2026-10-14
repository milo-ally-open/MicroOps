# reduce_sum 实现任务

## 基础语义与工程入口

- [x] 固定一维 float32、空输入为零和 CPU null-pointer 错误语义。
- [x] 建立 CPU 参考实现及无第三方测试程序。
- [x] 建立 CPU-only 默认 CMake 配置和独立后端选项。
- [x] 编写规格、使用说明、灰度均值示例和打包入口。
- [x] 标记未验证后端状态，不填写虚构性能结果。
- [ ] 在具备 CMake/C++17 的环境中构建并运行 CPU 测试。

## CUDA

- [x] 建立 kernel、host 和独立 CMake 文件位置。
- [x] 提供候选 float32 归约实现并标记为未验证。
- [ ] 在支持的 CUDA Toolkit/编译器上构建。
- [ ] 在 CUDA 设备上运行边界和随机输入正确性测试。
- [ ] 将 CUDA 运行测试加入自动化，并记录设备与运行时版本。
- [ ] 通过验证后更新 `operator.yaml` 状态。

## Ascend C

- [x] 建立 kernel、host 和独立构建拒绝入口。
- [ ] 选择并记录目标 CANN/Ascend C API 版本。
- [ ] 实现 kernel、host launch/注册及正确性测试。
- [ ] 在 Ascend C 工具链和设备上构建、运行并记录环境。
- [ ] 通过验证后更新 `operator.yaml` 状态。

## 后续改进

- [ ] 比较串行、分块树形归约和不同输入规模的精度与吞吐。
- [ ] 扩展 dtype 或多轴语义前，先更新提案与规格。
- [ ] 为组合应用添加 showcase。
