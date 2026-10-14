# 算子开发 SOP

本文是新增或改造算子的唯一工作流，适用于逐元素算子、归约、卷积和 Attention。按同一顺序完成规格、实现、独立测试工程、资源与性能评估、打包和记录；不同算子只替换其规格、交付件和测试参数。仓库只维护 CUDA/CANN 设备实现，不维护 CPU 算子后端。

## 1. 检查工作区和后端环境

开始前查看 Git 状态，保留已有内容。所有工具链、SDK、设备和驱动都必须实际探测；不要因主机系统有工具而假设 WSL 或容器中也可用。

```bash
git status --short --branch
python3 --version
cmake --version
python3 -m pip install -r requirements.txt
```

按目标后端检查编译器和设备，例如 `nvcc --version`、`nvidia-smi`，或 CANN 编译工具链与 NPU 设备状态。未能确认的依赖记为不可用，不能以 scaffold 或配置占位冒充构建成功。

## 2. 定义算子契约

先在 `operators/<op>/docs/` 明确数学定义、输入输出、布局、dtype、shape、合法范围、边界行为、精度/容差、别名规则、支持后端和验收标准，再开始实现。`operator.yaml` 集中记录工具要读取的算子根目录、后端源码目录和默认文件、CMake 项目、测试文件、结果路径及打包清单；详细规格仍放在算子文档中。

`operators/<op>/docs/reference/` 用于论文和引用资料，`operators/<op>/docs/notes/` 用于学习与实现笔记；它们不是 CPU 参考实现目录。正确性期望值由测试工程内的独立校验逻辑提供，不形成 CPU 算子交付后端。

**add v1 契约示例：**连续 float32 向量满足 `C[i] = A[i] + B[i]`。输入输出 shape 为 `[N]`，不支持 stride/broadcast。`N=0` 不访问指针也不发射 kernel；非空指针、`N` 上限和 `1e-6` 验收容差写入 spec。

## 3. 建立算子目录和后端交付件

每个算子的设备源码放在 `src/cuda/` 和 `src/cann/`，按需建立 `kernel/`、`host/`、`include/` 和后端构建文件；独立的后端测试工程放在 `tests/cuda/`、`tests/cann/`。正确性、内存和性能评估及其结果都归入对应测试工程，例如 `tests/cuda/performance/results/`。示例按后端放在 `examples/cuda/`、`examples/cann/`，动画源码放在 `animation/`；提案、规格、任务、用法、笔记和参考资料统一放在 `docs/` 下。

CUDA 与 CANN 的 kernel/host 是各自后端的真实交付件。一个后端缺少实现时，相关源码和测试目录只标记为 scaffold，构建和测试入口须明确失败或不可运行。构建产物、测试临时文件与动画视频只放在忽略的 `build/`、`dist/` 或媒体输出目录。

kernel 和 host 文件名由开发者决定，不要求采用固定的版本编号格式。CUDA 与 CANN 有各自的工具入口：`tools/operator_cuda.py` 只操作 CUDA，`tools/operator_cann.py` 只操作 CANN；`list` 用于查看候选文件。执行 build、test、benchmark 和 package 时，可分别用 `--kernel <文件名>`、`--host <文件名>` 指定源码。某一目录只有一个候选文件时可省略对应参数；存在多个候选时必须显式选择。构建输出按所选文件组合隔离。两入口共享的配置解析和通用流程位于 `tools/operator_common.py`。

**add 示例：**v1 是用户给出的单线程单元素 kernel；v2 是对齐时每线程用 `float4` 处理四个元素、未对齐时回退标量 grid-stride 的候选。两者都使用同一 host API 和测试工程。CANN v1/v2 当前已在 Atlas 200I DK A2（Ascend 310B4）和 CANN 7.0 RC1 上验证；v1 是标量循环，v2 每轮展开处理四个元素以尝试增加指令级并行，不声称使用 SIMD 指令或在其他 SoC/SDK 上可用。

## 4. 实现后端并建立测试工程

每个后端的 `tests/<backend>/CMakeLists.txt` 都是独立测试项目，使用同一组选择参数构建对应的 kernel/host，不复制另一份待测源码。正确性和性能测试使用同一组选择，结果记录具体文件名。测试工程应按职责组织：

- `correctness/`：典型 shape、空/最小输入、尾部或非对齐 shape、非法参数、容差和输出校验。
- `memory/`：理论工作集、实际申请字节、设备总量、分配前/分配后/释放后可用量、申请状态；必要时覆盖 workspace、临时张量和峰值估算。
- `performance/`：warmup、多轮设备计时、延迟、吞吐、有效带宽或 FLOP/s 等适用指标。

正确性通过和资源/性能评估是不同结论：benchmark 不代替完整正确性测试；构建成功也不证明设备结果正确。缺少编译器时配置需明确失败；缺少设备时测试标记 Skip 并说明原因，Skip 不能记为通过。

**add v1：**CUDA correctness 覆盖 `N=0`、`N=1`、`N=259`、负数长度和非法空指针。CUDA memory 工程申请 A/B/C 三个缓冲并记录实际设备内存读数；performance 工程使用 CUDA Event 测量 kernel 并校验输出样本。

## 5. 构建并运行测试

使用对应后端入口选择算子，先构建交付件，再运行独立测试工程：

```bash
python3 tools/operator_cuda.py list operators/add/operator.yaml
python3 tools/operator_cuda.py build operators/add/operator.yaml
python3 tools/operator_cuda.py test operators/add/operator.yaml
```

CUDA/CANN 入口分别固定后端，不再传 `--backend`。默认 kernel 和 host 从配置读取；必要时用 `--kernel`、`--host` 覆盖，不需要把源码路径重复写进每条命令。CANN 流程使用 `tools/operator_cann.py` 执行相同子命令。

可以直接对算子根目录运行 CMake，也可以独立配置测试工程；具体选项由对应 CMake 文件和算子 `docs/usage.md` 维护。使用分开的构建目录，记录 configure、compile、test 三个阶段和工具链版本。关闭或未选择的后端不应成为已验证状态。

**add：**CUDA Toolkit、CMake 3.21+ 和 C++17 编译器就绪后，命令按指定 kernel/host 文件构建算子库和 `tests/cuda/` 测试工程。没有 `nvcc` 时配置明确失败，也不声称后端通过。

## 6. 评估内存和性能

先按 dtype/shape 计算各输入、输出、中间张量、workspace 和临时缓冲的理论需求，标明估算口径。再在不耗尽共享设备的前提下逐步扩大规模，记录实际分配结果；OOM 是该规模不能分配，不是成功。内存报告注明设备总量、分配前/后/释放后读数和申请量。内存占用估算不能替代运行时观测，也不能将输入输出总字节直接称为峰值显存。

性能只在正确性验证后运行。记录设备型号、shape/元素数、dtype、编译器及版本、驱动/runtime/SDK、构建选项、warmup、重复次数、计时范围、延迟、校验状态和适用的吞吐指标。有效带宽/FLOP/s 必须说明字节或操作数的计算方式；估算有效带宽不是硬件计数器读出的 DRAM 带宽。比较加速比时基线须为相同输入、环境和计时口径。

没有实际运行的数据留空。每条记录保存在对应测试 suite 的 `tests/<backend>/<suite>/results/`，不得编造性能数字或把历史数据改成当前环境结果。

**add v1：**三块 float32 向量的理论工作集为 `3 * N * 4 = 12N` 字节。性能工具按 A/B 读取和 C 写回估算每轮流量 `12N`，使用设备计时器测 kernel，并将运行时、显存读数、延迟、估算有效带宽和校验状态写入 CSV。

运行 add CUDA 性能评估：

```bash
python3 tools/operator_cuda.py benchmark operators/add/operator.yaml
```

输入规模和迭代次数由 `operator.yaml` 的 `backends.<backend>.benchmark.sizes` 与 `iterations` 设置，不在常用命令里重复填写。`sizes` 的单位由该算子的 benchmark 定义，并在 `spec.md`/`usage.md` 说明；add 的单位是向量元素个数，不是字节数。add 当前配置为 `[1024, 1048576, 16777216]`，每个规模迭代 `100` 次。需要调整常用测量范围时改配置；临时单次测量仍可用 `--sizes` 和 `--iterations` 覆盖。对比另一组源码时，用 `--kernel` 和 `--host` 覆盖配置中的默认选择。

## 7. 更新文档、打包和提交

同步更新算子的 `proposal.md`、`spec.md`、`tasks.md`、`usage.md`、`operator.yaml`、后端构建配置、测试工程和工具登记。README、构建文档和 SOP 只描述当前实际存在的路径与状态。

打包命令只接受确实实现并登记的后端：

```bash
python3 tools/operator_cuda.py package operators/add/operator.yaml
python3 tools/operator_cann.py package operators/add/operator.yaml
```

包放入 `dist/`，包括所选后端交付件、测试工程、必要元数据及使用说明。源码包不等同于已编译二进制包。提交前运行可用的正确性测试、检查包文件清单和 `git diff --check`，并分别汇报 PASS、FAIL、SKIP 与未能验证的部分。

**add 当前状态：**v1/v2 已在 RTX 4060 Laptop GPU、nvcc 12.6、CUDA Runtime 12.6 上分别构建并通过 correctness 与 memory suites，并按相同输入规模和迭代数完成一轮 benchmark。新记录按 kernel/host 文件组合保存在独立 CSV；单轮性能数据不代表普遍结论。CANN v1/v2 已在 Atlas 200I DK A2 / Ascend 310B4、CANN 7.0 RC1（dav-m300）上构建并分别通过 correctness 和设备内存评估；benchmark 按 kernel/host 组合记录事件计时、编译器/runtime、设备资源与校验状态。单轮结果不构成普遍性能结论；实测仅适用于记录的板卡与工具链。
