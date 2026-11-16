import torch
from torch.utils.benchmark import Timer
from torch.utils.cpp_extension import load_inline
from pathlib import Path

cuda = Path(__file__).with_name("solve.cu").read_text() + r"""
void run(torch::Tensor image) {
    solve(image.data_ptr<unsigned char>(), image.size(1), image.size(0));
}
"""

custom = load_inline(
    name="custom_color_inversion",
    cpp_sources="void run(torch::Tensor);",
    cuda_sources=cuda,
    functions="run",
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
).run

def pytorch_inversion(image): 
    image[..., :3].bitwise_not_()

if __name__ == "__main__":

    # test
    width, height = 1025, 2049
    image = torch.randint(0, 256, (height, width, 4), device="cuda", dtype=torch.uint8)
    image_clone = image.clone()
    image_clone[..., :3] = 255 - image_clone[..., :3]
    custom(image)
    torch.testing.assert_close(image, image_clone, rtol=0, atol=0)
    print("test sucessfully!")

    # bench
    for stmt in ("custom(image)", "pytorch_inversion(image)"):
        us = Timer(stmt, globals=globals()).blocked_autorange().median * 1e6 
        print(f"{stmt}: {us:.3f} us")
