


# 向量加法

给定两个长度均为 $N$ 的32位浮点向量 $A$ 和 $B$, 在 NPU 上对它们进行逐元素加法, 将结果写入向量 $C$:

$$
C[i] = A[i] + B[i], \space 0 \le i \le N
$$
接口:

```cpp
void solve(const float* A, const float* B, float* C, int N); 
```  

所有指针参数均指向NPU设备内存, 数组由调用方式分配, 尺寸参数和其他标量按值插入。

| 参数 | 含义 |
| :--- | :--- |
| A | 长度为 N 的只读 float 输入向量。 |
| B | 长度为 N 的只读 float 输入向量。 |
| C | 长度为 N 的 float 输出向量，写入 A + B。 |
| N | 两个输入向量的元素个数。 |

## CANN 构建与运行

在配置好与开发板匹配的 CANN 环境后，从仓库根目录运行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 ops-from-scratch/element-wise/01-vector-add/cann/bench.py
```

脚本使用 NumPy 和 CANN 提供的 PyACL，不依赖 PyTorch 或 `torch_npu`。运行时会将 CANN kernel 编译为临时共享库，执行正确性测试和简单 benchmark，退出后清理临时编译产物。测试结果会显示 NPU 型号、驱动、编译器、输入规模、设备内存大小、耗时和估算带宽。当前 kernel 按 `dav-c220` 编译，已在 Atlas 200I DK A2（310B4，CANN 7.0.RC1）验证；其他 SoC 和 CANN 版本尚未验证。
