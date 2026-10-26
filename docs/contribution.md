# 贡献指南

1. 修改前先检查 Git 状态，保留已有文件与用户改动。
2. 为新算子或语义变更同步更新该算子的 proposal、spec、tasks、usage 和 operator.yaml。
3. test、bench、profile 按三个独立阶段说明结果；用统一入口 `python3 tools/run_operator.py <test|bench|profile> ... --backend <...>`。
4. 记录实际验证的设备、shape、dtype、工具链/运行时、源码选择和结果；明确区分 PASS、FAIL、SKIP 与未验证。
5. bench 只写入真实采集值；profile 产物写入 build/，不可把 profiler 期间的时间冒充 bench 延迟。
6. CUDA/CANN/PyTorch/Triton 状态以 operator.yaml 为准。未在对应设备实测的实现保持 scaffold/source_only。
7. 动画、构建、profile trace 和临时数据不得混入源码目录。
