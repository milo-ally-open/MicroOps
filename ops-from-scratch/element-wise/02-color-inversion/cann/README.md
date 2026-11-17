# 颜色反转

给定一张宽为 $W$、高为 $H$ 的 RGBA 图像，在 NPU 上原地反转 RGB 三个颜色通道，并保留 alpha 通道：

$$
I[y, x, c] = 255 - I[y, x, c],\quad c \in \{R,G,B\};\qquad I[y,x,A] \text{ 保持不变}
$$

接口：

```cpp
void solve(unsigned char* image, int width, int height);
```

`image` 指向 NPU 设备内存，数据按连续的 RGBA 顺序存储，每个像素占 4 个 `unsigned char`。算子会原地修改 RGB，保留 alpha；调用前由调用方分配设备内存。

| 参数 | 含义 |
| :--- | :--- |
| `image` | 连续、按 RGBA 排列的设备端图像缓冲区，大小为 `height * width * 4` 字节。 |
| `width` | 图像宽度，单位为像素。 |
| `height` | 图像高度，单位为像素。 |

当前 CANN kernel 按像素线性遍历连续 RGBA 缓冲区，逐通道反转 RGB 并保留 alpha。输入形状为 `(height, width, 4)`，宽高应为正数。



## CANN 构建与运行

配置好与开发板匹配的 CANN 环境后，进入本目录运行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python bench.py
```

脚本使用 NumPy 和 CANN 提供的 PyACL，不依赖 PyTorch 或 `torch_npu`。运行时会将 CANN kernel 编译为临时共享库，执行正确性测试和简单 benchmark，退出后清理临时编译产物。测试结果会显示 NPU 型号、驱动、编译器、输入规模、设备内存大小、耗时和估算带宽。当前 kernel 按 `dav-c220` 编译，已在 Atlas 200I DK A2（310B4，CANN 7.0.RC1）验证；其他 SoC 和 CANN 版本尚未验证。
