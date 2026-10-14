# add 实现任务

- [x] 明确 float32 一维向量加法的共同语义和边界。
- [x] 建立 CUDA v1 标量 kernel 与 host launch wrapper。
- [x] 建立 CUDA v2 向量化候选与 host launch wrapper。
- [x] 建立 CUDA 可选源码构建、correctness、显存评估和 benchmark。
- [x] 在 RTX 4060 Laptop GPU 上验证 CUDA 源码组合并记录真实测量。
- [x] 建立 CANN v1 标量 CCE kernel/host 和 CMake/bisheng 构建入口。
- [x] 建立 CANN v2 四元素展开 kernel/host，并使用同一测试工程进行独立选择验证。
- [x] 建立 CANN correctness（N=0/1/259/1027、非法参数）、设备内存评估和事件 benchmark。
- [x] 分别记录 CANN v1/v2 的真实 benchmark 数据，不预设展开版本必然更快。
- [x] 在 Atlas 200I DK A2 / Ascend 310B4、CANN 7.0 RC1（dav-m300）板端构建、测试和记录。
- [ ] 在其他受支持 CANN 版本/SoC 上验证或升级到相应的高层 Ascend C vector API。
- [ ] 通过同一设备、同一规模、同一测试条件比较后续 CANN 优化实现。
