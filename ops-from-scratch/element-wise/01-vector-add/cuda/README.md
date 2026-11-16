


# 向量加法

给定两个长度均为 $N$ 的32位浮点向量 $A$ 和 $B$, 在 GPU 上对它们进行逐元素加法, 将结果写入向量 $C$:

$$
C[i] = A[i] + B[i], \space 0 \le i \le N
$$
接口:

```cpp
void solve(const float* A, const float* B, float* C, int N); 
```  

所有指针参数均指向GPU设备内存, 数组由调用方式分配, 尺寸参数和其他标量按值插入。

| 参数 | 含义 |
| :--- | :--- |
| A | 长度为 N 的只读 float 输入向量。 |
| B | 长度为 N 的只读 float 输入向量。 |
| C | 长度为 N 的 float 输出向量，写入 A + B。 |
| N | 两个输入向量的元素个数。 |