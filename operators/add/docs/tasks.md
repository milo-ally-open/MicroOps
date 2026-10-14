# add 实现任务

- [x] 明确 float32 一维向量加法与 CUDA v1 范围。
- [x] 建立 CUDA v1 kernel：全局索引、尾部保护、单线程单元素。
- [x] 建立 CUDA v2 向量化候选：对齐时 float4 处理，未对齐时标量回退。
- [x] 建立两组可分别选择的 CUDA host launch wrapper 和算子级构建入口。
- [x] 建立可按 kernel/host 文件组合选择的 CUDA 正确性测试，包含边界、未对齐指针和非法参数检查。
- [x] 建立 CUDA 显存评估与性能评估程序及真实历史测量记录。
- [x] 完成独立 CUDA 测试工程配置，并在 nvcc 12.6 工具链上构建运行。
- [x] 完成 CUDA memory suite 的设备运行并检查生成记录。
- [ ] 实现 CANN kernel、host 调度和对应测试工程；当前目录仅为 scaffold。
- [x] 分别构建并运行两组 kernel/host 文件的 correctness 与 memory suites。
- [x] 使用相同设备、shape 和计时条件实测两组实现；只记录真实结果，不预设向量化实现更快。
- [x] 为已验证的源码组合补充设备、工具链、资源与性能记录。
