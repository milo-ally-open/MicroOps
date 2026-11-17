import time
_script_start = time.perf_counter()
print("[start] vector add CUDA benchmark; importing PyTorch", flush=True)

import torch
from torch.utils.benchmark import Timer
from torch.utils.cpp_extension import load_inline
from pathlib import Path

cuda = Path(__file__).with_name("solve.cu").read_text() + r"""
void run(torch::Tensor A, torch::Tensor B, torch::Tensor out) {
    solve(A.data_ptr<float>(), B.data_ptr<float>(), out.data_ptr<float>(), A.numel());
}
"""

print("[compile] building/loading CUDA extension (first run may take a while)", flush=True)
_compile_start = time.perf_counter()
custom = load_inline(
    name="custom_vector_add",
    cpp_sources="void run(torch::Tensor, torch::Tensor, torch::Tensor);",
    cuda_sources=cuda,
    functions="run",
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    verbose=True,
).run
print(f"[compile] ready in {time.perf_counter() - _compile_start:.2f}s", flush=True)

if __name__ == "__main__":
    print(f"[environment] PyTorch={torch.__version__}; CUDA={torch.version.cuda}; device={torch.cuda.get_device_name()}", flush=True)
    print("[test] running correctness check", flush=True)
    a = torch.rand(2**10 + 3, device="cuda", dtype=torch.float32)
    b, out = torch.rand_like(a), torch.empty_like(a)
    custom(a, b, out)
    torch.testing.assert_close(out, a + b)
    print("[test] passed", flush=True)

    # bench 
    for stmt in ("custom(a, b, out)", "torch.add(a, b, out=out)"):
        print(f"[bench] measuring {stmt} (warmup and adaptive timing; min 0.2s)", flush=True)
        _bench_start = time.perf_counter()
        us = Timer(stmt=stmt, globals=globals()).blocked_autorange(min_run_time=0.2).median * 1e6
        print(f"[bench] {stmt}: {us:.3f} us; completed in {time.perf_counter() - _bench_start:.2f}s", flush=True)
    print(f"[done] total elapsed {time.perf_counter() - _script_start:.2f}s", flush=True)
