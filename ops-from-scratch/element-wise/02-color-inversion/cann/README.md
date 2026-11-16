# 颜色反转

给定一张宽为 $W$、高为 $H$ 的 RGBA 图像，在 GPU 上原地反转 RGB 三个颜色通道，并保留 alpha 通道：

$$
I[y, x, c] = 255 - I[y, x, c],\quad c \in \{R,G,B\};\qquad I[y,x,A] \text{ 保持不变}
$$

接口：

```cpp
void solve(unsigned char* image, int width, int height);
```

`image` 指向 GPU 设备内存，数据按连续的 RGBA 顺序存储，每个像素占 4 个 `unsigned char`。算子会原地修改 RGB，保留 alpha；调用前由调用方分配设备内存。

| 参数 | 含义 |
| :--- | :--- |
| `image` | 连续、按 RGBA 排列的设备端图像缓冲区，大小为 `height * width * 4` 字节。 |
| `width` | 图像宽度，单位为像素。 |
| `height` | 图像高度，单位为像素。 |

当前 kernel 将每个 RGBA 像素作为一个 `uchar4`，由一个 CUDA 线程按网格步长处理一个或多个像素。输入张量形状为 `(height, width, 4)`。宽高应为正数，缓冲区需要满足 `uchar4` 对齐要求。


