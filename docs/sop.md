# 算子开发 SOP

本文是新增或改造算子的唯一工作流，适用于逐元素算子、归约、卷积和 Attention。按同一顺序完成规格、实现、独立测试工程、资源与性能评估、打包和记录；不同算子只替换其规格、交付件和测试参数。仓库只维护 CUDA/CANN 设备实现，不维护 CPU 算子后端。

## 1. 检查工作区和后端环境

开始前查看 Git 状态，保留已有内容。所有工具链、SDK、设备和驱动都必须实际探测；不要因主机系统有工具而假设 WSL 或容器中也可用。

```bash
git status --short --branch
python3 --version
cmake --version
```

按目标后端检查编译器和设备，例如 `nvcc --version`、`nvidia-smi`，或 CANN 编译工具链与 NPU 设备状态。未能确认的依赖记为不可用，不能以 scaffold 或配置占位冒充构建成功。

## 2. 定义算子契约

先在 `operators/<op>/docs/` 明确数学定义、输入输出、布局、dtype、shape、合法范围、边界行为、精度/容差、别名规则、支持后端和验收标准，再开始实现。`operator.yaml` 只登记工具需要的版本、后端状态、源文件、测试工程和 demo 入口，不复制完整规格。

`operators/<op>/docs/reference/` 用于论文和引用资料，`operators/<op>/docs/notes/` 用于学习与实现笔记；它们不是 CPU 参考实现目录。正确性期望值由测试工程内的独立校验逻辑提供，不形成 CPU 算子交付后端。

**add v1 契约示例：**连续 float32 向量满足 `C[i] = A[i] + B[i]`。输入输出 shape 为 `[N]`，不支持 stride/broadcast。`N=0` 不访问指针也不发射 kernel；非空指针、`N` 上限和 `1e-6` 验收容差写入 spec。

## 3. 建立算子目录和后端交付件

每个算子的设备源码放在 `src/cuda/` 和 `src/cann/`，按需建立 `kernel/`、`host/`、`include/` 和后端构建文件；独立的后端测试工程放在 `tests/cuda/`、`tests/cann/`。正确性、内存和性能评估及其结果都归入对应测试工程，例如 `tests/cuda/performance/results/`。示例按后端放在 `examples/cuda/`、`examples/cann/`，动画源码放在 `animation/`；提案、规格、任务、用法、笔记和参考资料统一放在 `docs/` 下。

CUDA 与 CANN 的 kernel/host 是各自后端的真实交付件。一个后端缺少实现时，相关源码和测试目录只标记为 scaffold，构建和测试入口须明确失败或不可运行。构建产物、测试临时文件与动画视频只放在忽略的 `build/`、`dist/` 或媒体输出目录。

**add v1：**`src/cuda/kernel/add_kernel.cu` 是用户给出的单线程单元素 kernel；`src/cuda/host/add_cuda.cu` 校验参数、计算 launch grid 并在调用方 stream 上启动它。`src/cann/` 只保留 scaffold，不声称实现了 add。

## 4. 实现后端并建立测试工程

每个后端的 `tests/<backend>/CMakeLists.txt` 都是独立测试项目，直接引用同级 `<backend>/` 的交付件，不复制另一份待测 kernel。测试工程应按职责组织：

- `correctness/`：典型 shape、空/最小输入、尾部或非对齐 shape、非法参数、容差和输出校验。
- `memory/`：理论工作集、实际申请字节、设备总量、分配前/分配后/释放后可用量、申请状态；必要时覆盖 workspace、临时张量和峰值估算。
- `performance/`：warmup、多轮设备计时、延迟、吞吐、有效带宽或 FLOP/s 等适用指标。

正确性通过和资源/性能评估是不同结论：benchmark 不代替完整正确性测试；构建成功也不证明设备结果正确。缺少编译器时配置需明确失败；缺少设备时测试标记 Skip 并说明原因，Skip 不能记为通过。

**add v1：**CUDA correctness 覆盖 `N=0`、`N=1`、`N=259`、负数长度和非法空指针。CUDA memory 工程申请 A/B/C 三个缓冲并记录实际设备内存读数；performance 工程使用 CUDA Event 测量 kernel 并校验输出样本。

## 5. 构建并运行测试

按同一入口选择算子和后端，先构建交付件，再运行独立测试工程：

```bash
python3 tools/ops.py build --operator add --backend cuda
python3 tools/ops.py test --operator add --backend cuda
```

可以直接对算子根目录运行 CMake，也可以独立配置测试工程；具体选项由对应 CMake 文件和算子 `docs/usage.md` 维护。使用分开的构建目录，记录 configure、compile、test 三个阶段和工具链版本。关闭或未选择的后端不应成为已验证状态。

**add v1：**CUDA Toolkit、CMake 3.21+ 和 C++17 编译器就绪后，以上命令会分别构建算子库和 `tests/cuda/` 测试工程。没有 `nvcc` 时不进入设备测试，也不声称后端通过。

## 6. 评估内存和性能

先按 dtype/shape 计算各输入、输出、中间张量、workspace 和临时缓冲的理论需求，标明估算口径。再在不耗尽共享设备的前提下逐步扩大规模，记录实际分配结果；OOM 是该规模不能分配，不是成功。内存报告注明设备总量、分配前/后/释放后读数和申请量。内存占用估算不能替代运行时观测，也不能将输入输出总字节直接称为峰值显存。

性能只在正确性验证后运行。记录设备型号、shape/元素数、dtype、编译器及版本、驱动/runtime/SDK、构建选项、warmup、重复次数、计时范围、延迟、校验状态和适用的吞吐指标。有效带宽/FLOP/s 必须说明字节或操作数的计算方式；估算有效带宽不是硬件计数器读出的 DRAM 带宽。比较加速比时基线须为相同输入、环境和计时口径。

没有实际运行的数据留空。每条记录保存在对应测试 suite 的 `tests/<backend>/<suite>/results/`，不得编造性能数字或把历史数据改成当前环境结果。

**add v1：**三块 float32 向量的理论工作集为 `3 * N * 4 = 12N` 字节。性能工具按 A/B 读取和 C 写回估算每轮流量 `12N`，使用设备计时器测 kernel，并将运行时、显存读数、延迟、估算有效带宽和校验状态写入 CSV。

运行 add CUDA 性能评估：

```bash
python3 tools/ops.py benchmark --operator add \
  --sizes 1024,1048576,16777216 --iterations 100
```

## 7. 更新文档、打包和提交

同步更新算子的 `proposal.md`、`spec.md`、`tasks.md`、`usage.md`、`operator.yaml`、后端构建配置、测试工程和工具登记。README、构建文档和 SOP 只描述当前实际存在的路径与状态。

打包命令只接受确实实现并登记的后端：

```bash
python3 tools/ops.py package --operator add --backend cuda
```

包放入 `dist/`，包括所选后端交付件、测试工程、必要元数据及使用说明。源码包不等同于已编译二进制包。提交前运行可用的正确性测试、检查包文件清单和 `git diff --check`，并分别汇报 PASS、FAIL、SKIP 与未能验证的部分。

**add v1 当前状态：**历史 CUDA 性能 CSV 保留为已有实测记录，位于 `tests/cuda/performance/results/`；新的根目录布局和测试工程待本机 CUDA 工具链复验；CANN 实现及 CANN 测试工程仍是 scaffold。
