# mul 实现任务

- [x] 定义算子语义、输入约束、dtype、shape、精度与边界测试。
- [x] 建立 CUDA v1/v2 kernel、host wrapper、独立构建入口和测试工程。
- [x] 在 RTX 4060 Laptop GPU 完成 CUDA build、test、memory 与 bench；保存真实记录。
- [x] PyTorch 与 Triton GPU 实现完成 test、bench、torch.profiler profile。
- [x] CANN v1/v2 已在 Atlas 200I DK A2 / CANN 7.0.RC1 完成 build、test、memory、bench；已采集可用的 msprof activity。
- [ ] 在支持当前硬件的更新版 Ascend C SDK 上验证 vector 路径，并做严格配对调优。
- [x] 建立配置元数据、demo/使用说明和按算子组织的测试记录。
- [x] 整理统一 test / bench / profile 工作流与本轮性能分析。
- [ ] 在无背景负载下重复配对 CUDA/PyTorch/Triton 测量并报告统计区间；CUDA NCU 需先解决 WSL counter 权限。
