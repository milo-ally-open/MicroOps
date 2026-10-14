# add 实现任务

- [x] 明确 float32 一维向量加法与 CUDA v1 范围。
- [x] 建立 CUDA v1 kernel：全局索引、尾部保护、单线程单元素。
- [x] 建立 CUDA host launch wrapper 和算子级构建入口。
- [x] 建立 CUDA 正确性测试源文件，包含边界和非法参数检查。
- [x] 建立 CUDA 显存评估与性能评估程序及真实历史测量记录。
- [ ] 完成新的独立 CUDA 测试工程配置，并在工具链可用时重新构建运行。
- [ ] 完成 CUDA memory suite 的设备运行并检查生成记录。
- [ ] 实现 CANN kernel、host 调度和对应测试工程；当前目录仅为 scaffold。
- [ ] 为新版本补充经实际运行的设备、工具链、资源与性能记录。
