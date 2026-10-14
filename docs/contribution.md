# 贡献约定

1. 按[算子开发 SOP](sop.md) 完成规格、实现、测试工程、评估和打包。
2. CUDA/CANN 实现分别位于 `operators/<op>/src/cuda/` 和 `operators/<op>/src/cann/`；测试工程放在 `operators/<op>/tests/<backend>/`，并引用被测交付件。
3. correctness、memory、performance 的结果分开说明。记录设备、shape、dtype、工具链、运行时、测量口径和校验状态。
4. 后端未在目标工具链及设备上验证时应标为 scaffold/未验证；Skip 不得记为通过。
5. benchmark 只记录真实测量；未知值留空。生成视频、构建文件和临时产物不放进源码目录。
6. 算子文档维护在该算子的 `docs/`，项目通用流程维护在仓库 `docs/`。
7. 不在没有明确许可的情况下添加许可证。
