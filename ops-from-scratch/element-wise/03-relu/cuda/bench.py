from pathlib import Path

import torch
from torch.utils.benchmark import Timer
from torch.utils.cpp_extension import load_inline

cuda = Path(__file__).with_name("solve.cu").read_text() + r"""
void run(torch::Tensor input, torch::Tensor output) {
    if (input.numel() == 0) return;
    solve(input.data_ptr<float>(), output.data_ptr<float>(), input.numel());
}
"""

print("Building/loading CUDA extension...", flush=True)
custom = load_inline(
    name="custom_relu",
    cpp_sources="void run(torch::Tensor, torch::Tensor);",
    cuda_sources=cuda,
    functions="run",
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
).run
print("CUDA extension ready.", flush=True)

if __name__ == "__main__":
    print(f"GPU={torch.cuda.get_device_name()}; PyTorch={torch.__version__}; CUDA={torch.version.cuda}")
    print("Testing correctness...", flush=True)
    for n in (0, 1, 3, 4, 5, 257, 4099):
        input = (torch.arange(n, device="cuda", dtype=torch.float32) % 11) - 5
        output = torch.empty_like(input)
        custom(input, output)
        torch.testing.assert_close(output, torch.relu(input), rtol=0, atol=0)
    print("Tests passed (N=0, 1, 3, 4, 5, 257, 4099).")

    n = 2**20 + 3
    input = (torch.arange(n, device="cuda", dtype=torch.float32) % 257) - 128
    output = torch.empty_like(input)
    print(f"Benchmark results (N={n}, dtype=float32):")
    for stmt in ("custom(input, output)", "torch.clamp(input, min=0, out=output)"):
        us = Timer(stmt, globals=globals()).blocked_autorange(min_run_time=0.2).median * 1e6
        print(f"  {stmt}: {us:.3f} us")
