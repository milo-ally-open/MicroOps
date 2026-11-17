import torch
from torch.utils.benchmark import Timer
from torch.utils.cpp_extension import load_inline
from pathlib import Path

cuda = Path(__file__).with_name("solve.cu").read_text() + r"""
void run(torch::Tensor A, torch::Tensor B, torch::Tensor out) {
    solve(A.data_ptr<float>(), B.data_ptr<float>(), out.data_ptr<float>(), A.numel());
}
"""

print("Building/loading CUDA extension...", flush=True)
custom = load_inline(
    name="custom_vector_add",
    cpp_sources="void run(torch::Tensor, torch::Tensor, torch::Tensor);",
    cuda_sources=cuda,
    functions="run",
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
).run
print("CUDA extension ready.", flush=True)

if __name__ == "__main__":
    print(f"GPU={torch.cuda.get_device_name()}; PyTorch={torch.__version__}; CUDA={torch.version.cuda}")
    print("Testing correctness...", flush=True)
    a = torch.rand(2**10 + 3, device="cuda", dtype=torch.float32)
    b, out = torch.rand_like(a), torch.empty_like(a)
    custom(a, b, out)
    torch.testing.assert_close(out, a + b)
    print("Tests passed.")

    print(f"Benchmark results (N={a.numel()}, dtype=float32):")
    for stmt in ("custom(a, b, out)", "torch.add(a, b, out=out)"):
        us = Timer(stmt=stmt, globals=globals()).blocked_autorange(min_run_time=0.2).median * 1e6
        print(f"  {stmt}: {us:.3f} us")
