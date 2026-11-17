# ReLU

给定长度为 $N$ 的 32 位浮点向量 $X$，在 GPU 上逐元素计算 ReLU，并将结果写入 $Y$：

$$
Y[i] = \max(0, X[i]),\quad 0 \le i < N
$$

接口：

```cpp
void solve(const float* input, float* output, int N);
```

`input` 和 `output` 均指向 GPU 设备内存，由调用方分配。输入按连续的一维 `float32` 向量处理；本练习使用独立的输入、输出缓冲区。

| 参数 | 含义 |
| :--- | :--- |
| `input` | 长度为 `N` 的只读 `float32` 输入向量。 |
| `output` | 长度为 `N` 的 `float32` 输出向量，写入 ReLU 结果。 |
| `N` | 向量元素个数；`N == 0` 时不访问输入或输出内存。 |

## 练习范围

- 每个输出元素对应一个 ReLU 结果：负数变为 `0`，正数保持不变，零仍为零。
- 先实现逐元素基础版本，并为任意 `N` 添加边界保护；重点检查长度不是线程块大小整数倍的情况。
- 测试使用有限的 `float32` 数值，覆盖负数、零、正数以及长度 `0`、`1` 和非整齐长度。NaN 和无穷值不属于本练习的验收输入。
- ReLU 对有限 `float32` 输入的结果应与参考结果逐元素一致。

## CUDA 构建与运行

需要可用的 CUDA Toolkit、`nvcc`，以及带 CUDA 支持的 PyTorch。完成 `solve.cu` 和 `bench.py` 后，进入本目录运行：

```bash
python bench.py
```

`bench.py` 将通过 PyTorch `load_inline` 编译同目录下的 `solve.cu`，执行正确性检查并比较自定义 CUDA 实现与 PyTorch ReLU 的运行时间。当前目录仅创建了练习文件和说明，kernel 与测试代码尚未实现，因此命令暂不可运行。
