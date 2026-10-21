# sub 实现任务

- [x] 从 `add` 工程结构建立 `sub` 的独立目录、配置和说明文档。
- [x] 写入 CUDA v1/v2 kernel 与对应 host 配对。
- [x] 写入 CANN v1/v2 kernel 与对应 host 配对。
- [x] 建立 CUDA/CANN correctness、设备内存评估、benchmark 和 demo 工程。
- [x] 将 CUDA 与 CANN 状态标记为 `scaffold`，不复制 `add` 的设备验证和 benchmark 记录。
- [ ] 在 CUDA Toolkit 主机上编译全部待验证的 kernel/host 组合并运行 correctness 与显存评估。
- [ ] 在目标 Ascend 板卡上用已安装 CANN SDK 编译全部待验证组合并运行 correctness 与设备内存评估。
- [ ] 分别实测 v1/v2 性能，确认记录包含设备、编译器/runtime、输入规模和有效带宽估算。
- [ ] 根据实际结果更新 `operator.yaml`、spec、usage 和 README；未验证的平台继续标注为 scaffold。
