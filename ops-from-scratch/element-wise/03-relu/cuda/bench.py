import time
from pathlib import Path

_script_start = time.perf_counter()
print("[start] ReLU CUDA benchmark; importing PyTorch", flush=True)

import torch
from torch.utils.benchmark import Timer
from torch.utils.cpp_extension import load_inline

cuda = Path(__file__).with_name("solve.cu").read_text() + r"""
void run(torch::Tensor input, torch::Tensor output) {
    if (input.numel() == 0) return;
    solve(input.data_ptr<float>(), output.data_ptr<float>(), input.numel());
}
"""

print("[compile] building/loading CUDA extension (first run may take a while)", flush=True)
_compile_start = time.perf_counter()
custom = load_inline(
    name="custom_relu",
    cpp_sources="void run(torch::Tensor, torch::Tensor);",
    cuda_sources=cuda,
    functions="run",
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    verbose=True,
).run
print(f"[compile] ready in {time.perf_counter() - _compile_start:.2f}s", flush=True)

if __name__ == "__main__":
    print(
        f"[environment] PyTorch={torch.__version__}; CUDA={torch.version.cuda}; "
        f"device={torch.cuda.get_device_name()}",
        flush=True,
    )

    print("[test] checking empty, short, vectorized, and tail sizes", flush=True)
    for n in (0, 1, 3, 4, 5, 257, 4099):
        print(f"[test] checking N={n}", flush=True)
        input = (torch.arange(n, device="cuda", dtype=torch.float32) % 11) - 5
        output = torch.empty_like(input)
        custom(input, output)
        torch.testing.assert_close(output, torch.relu(input), rtol=0, atol=0)
    print("[test] passed sizes: 0, 1, 3, 4, 5, 257, 4099", flush=True)

    n = 2**20 + 3
    input = (torch.arange(n, device="cuda", dtype=torch.float32) % 257) - 128
    output = torch.empty_like(input)
    print(f"[bench] input: N={n}; dtype=float32; bytes per tensor={input.numel() * input.element_size()}", flush=True)
    for stmt in ("custom(input, output)", "torch.clamp(input, min=0, out=output)"):
        print(f"[bench] measuring {stmt} (warmup and adaptive timing; min 0.2s)", flush=True)
        _bench_start = time.perf_counter()
        us = Timer(stmt, globals=globals()).blocked_autorange(min_run_time=0.2).median * 1e6
        print(f"[bench] {stmt}: {us:.3f} us; completed in {time.perf_counter() - _bench_start:.2f}s", flush=True)
    print(f"[done] total elapsed {time.perf_counter() - _script_start:.2f}s", flush=True)
