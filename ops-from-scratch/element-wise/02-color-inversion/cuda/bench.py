import time
_script_start = time.perf_counter()
print("[start] color inversion CUDA benchmark; importing PyTorch", flush=True)

import torch
from torch.utils.benchmark import Timer
from torch.utils.cpp_extension import load_inline
from pathlib import Path

cuda = Path(__file__).with_name("solve.cu").read_text() + r"""
void run(torch::Tensor image) {
    solve(image.data_ptr<unsigned char>(), image.size(1), image.size(0));
}
"""

print("[compile] building/loading CUDA extension (first run may take a while)", flush=True)
_compile_start = time.perf_counter()
custom = load_inline(
    name="custom_color_inversion",
    cpp_sources="void run(torch::Tensor);",
    cuda_sources=cuda,
    functions="run",
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    verbose=True,
).run
print(f"[compile] ready in {time.perf_counter() - _compile_start:.2f}s", flush=True)

def pytorch_inversion(image): 
    image[..., :3].bitwise_not_()

if __name__ == "__main__":
    print(f"[environment] PyTorch={torch.__version__}; CUDA={torch.version.cuda}; device={torch.cuda.get_device_name()}", flush=True)
    print("[test] running correctness check", flush=True)
    width, height = 1025, 2049
    image = torch.randint(0, 256, (height, width, 4), device="cuda", dtype=torch.uint8)
    image_clone = image.clone()
    image_clone[..., :3] = 255 - image_clone[..., :3]
    custom(image)
    torch.testing.assert_close(image, image_clone, rtol=0, atol=0)
    print("[test] passed; alpha channel preserved", flush=True)

    # bench
    for stmt in ("custom(image)", "pytorch_inversion(image)"):
        print(f"[bench] measuring {stmt} (warmup and adaptive timing; min 0.2s)", flush=True)
        _bench_start = time.perf_counter()
        us = Timer(stmt, globals=globals()).blocked_autorange(min_run_time=0.2).median * 1e6
        print(f"[bench] {stmt}: {us:.3f} us; completed in {time.perf_counter() - _bench_start:.2f}s", flush=True)
    print(f"[done] total elapsed {time.perf_counter() - _script_start:.2f}s", flush=True)
