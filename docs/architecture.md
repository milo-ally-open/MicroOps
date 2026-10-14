# 仓库架构

## 按算子组织

每个 `operators/<name>/` 是独立交付单元，包含规格文档、`src/cuda/` 与 `src/cann/` 后端目录、独立测试工程、测试评估记录、示例和动画源码。后端源文件统一收在 `src/` 下。仓库不维护 CPU 算子实现。

## 测试工程

`tests/cuda/` 和 `tests/cann/` 分别配置各自的工具链和运行环境，并直接使用同级设备实现作为被测对象。测试工程按 correctness、memory 和 performance 组织，评估结果保存在对应 suite 的 `results/` 中。评估结果由设备实测产生并记录环境信息；Skip、未实现和仅编译均不等于设备通过。

## 参考资料

`operators/<name>/docs/reference/` 用于论文、引用和阅读资料，`operators/<name>/docs/notes/` 用于开发学习笔记；二者都不存放算子实现。

## 当前能力

`add` 的 CUDA kernel/host 与测试工程已有设备验证；CANN v1 已在 Atlas 200I DK A2 / Ascend 310B4、CANN 7.0 RC1（dav-m300）上构建并通过设备测试。CANN v1 是已验证板卡上的标量 CCE 学习实现，不代表其他 SoC/SDK 可用。具体状态以 `operators/add/operator.yaml` 为准。
