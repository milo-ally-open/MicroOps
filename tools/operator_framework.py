#!/usr/bin/env python3
"""Test, bench, and profile the GPU-only PyTorch/Triton implementations."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> tuple[dict, Path]:
    if yaml is None:
        raise RuntimeError("PyYAML is required. Install project requirements with: python -m pip install -r requirements.txt")
    config_path = path if path.is_absolute() else ROOT / path
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or "name" not in config:
        raise ValueError(f"Invalid operator config: {config_path}")
    return config, config_path.parent


def load_runner(op_root: Path, config: dict, backend: str):
    profile = config.get("frameworks", {}).get(backend)
    if not profile:
        raise ValueError(f"{config['name']} has no {backend} framework configuration")
    source = (op_root / profile["implementation"]).resolve()
    if op_root.resolve() not in source.parents or not source.is_file():
        raise ValueError(f"Invalid or missing {backend} implementation path: {source}")
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for GPU execution. Install a CUDA-enabled PyTorch build first.") from error
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU is available to PyTorch; this command never falls back to CPU.")
    if backend == "triton":
        try:
            import triton  # noqa: F401
        except ImportError as error:
            raise RuntimeError("Triton is required for this backend. Install Triton in the active Python environment.") from error
    spec = importlib.util.spec_from_file_location(f"ops_{config['name']}_{backend}", source)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.run, torch


def inputs_for(op: str, size: int, torch):
    x = torch.linspace(-5.0, 5.0, size, device="cuda", dtype=torch.float32)
    if op in {"add", "sub", "mul", "div"}:
        y = torch.linspace(0.5, 2.0, size, device="cuda", dtype=torch.float32)
        return (x, y)
    return (x,)


def reference(op: str, args, torch):
    import torch.nn.functional as F
    if op == "add": return torch.add(*args)
    if op == "sub": return torch.sub(*args)
    if op == "mul": return torch.mul(*args)
    if op == "div": return torch.div(*args)
    x = args[0]
    if op == "relu": return torch.where(x > 0, x, torch.where(torch.isnan(x), x, torch.zeros_like(x)))
    if op == "gelu": return F.gelu(x, approximate="tanh")
    if op == "silu": return F.silu(x)
    if op == "sigmoid": return torch.sigmoid(x)
    if op == "tanh": return torch.tanh(x)
    raise ValueError(f"Unknown operator: {op}")


def command_list(args):
    config, op_root = load_config(args.config)
    print(f"operator: {config['name']}")
    for backend, profile in config.get("frameworks", {}).items():
        print(f"{backend}: status={profile.get('status')} implementation={profile.get('implementation')} entry={profile.get('entry')}")
    return 0


def command_test(args):
    config, op_root = load_config(args.config)
    runner, torch = load_runner(op_root, config, args.backend)
    profile = config["frameworks"][args.backend]
    op = config["name"]
    rtol, atol = float(profile["rtol"]), float(profile["atol"])
    for size in (0, 1, 259, 1027):
        values = inputs_for(op, size, torch)
        actual = runner(*values)
        expected = reference(op, values, torch)
        if actual.device.type != "cuda" or actual.dtype != torch.float32 or tuple(actual.shape) != (size,):
            raise AssertionError(f"Invalid output contract for {op}/{args.backend}, N={size}")
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
        print(f"PASS {op}/{args.backend} N={size}")
    if op == "relu":
        special = torch.tensor([-1.0, 0.0, 1.0, float("nan")], device="cuda", dtype=torch.float32)
        actual = runner(special)
        if not torch.isnan(actual[-1]).item():
            raise AssertionError("ReLU must propagate NaN")
        print(f"PASS {op}/{args.backend} NaN propagation")
    torch.cuda.synchronize()
    print(f"GPU correctness checks passed: {op}/{args.backend}")
    return 0


def command_bench(args):
    config, op_root = load_config(args.config)
    runner, torch = load_runner(op_root, config, args.backend)
    profile = config["frameworks"][args.backend]
    op = config["name"]
    sizes = args.sizes or profile["bench"]["sizes"]
    warmup = args.warmup if args.warmup is not None else int(profile["bench"]["warmup"])
    iterations = args.iterations if args.iterations is not None else int(profile["bench"]["iterations"])
    if iterations < 1 or warmup < 0 or any(int(n) < 0 for n in sizes):
        raise ValueError("sizes and warmup must be non-negative; iterations must be positive")
    rows = []
    for size_value in sizes:
        size = int(size_value)
        values = inputs_for(op, size, torch)
        for _ in range(warmup):
            runner(*values)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            runner(*values)
        end.record()
        torch.cuda.synchronize()
        latency_ms = start.elapsed_time(end) / iterations
        peak = torch.cuda.max_memory_allocated()
        bytes_per_element = int(profile["bench"]["bytes_per_element"])
        traffic = size * bytes_per_element
        bandwidth = (traffic / (latency_ms * 1e6)) if latency_ms > 0 else math.nan
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "operator": op,
            "backend": args.backend,
            "implementation": profile["implementation"],
            "device": torch.cuda.get_device_name(torch.cuda.current_device()),
            "input_size": size,
            "dtype": "float32",
            "torch_version": torch.__version__,
            "triton_version": _triton_version(args.backend),
            "cuda_runtime": str(torch.version.cuda or "unknown"),
            "warmup": warmup,
            "iterations": iterations,
            "peak_allocated_bytes": peak,
            "estimated_bytes_per_iteration": traffic,
            "average_latency_us": latency_ms * 1000.0,
            "effective_bandwidth_GBps": bandwidth,
        }
        rows.append(row)
        print(f"N={size}: {row['average_latency_us']:.3f} us, estimated {bandwidth:.3f} GB/s")
    result_dir = op_root / profile["bench"]["results"]
    result_dir.mkdir(parents=True, exist_ok=True)
    configured_output = Path(args.output) if args.output is not None else None
    result_file = configured_output or (result_dir / (Path(profile["implementation"]).stem + ".csv"))
    if not result_file.is_absolute():
        result_file = ROOT / result_file
    fields = list(rows[0]) if rows else []
    write_header = not result_file.exists() or result_file.stat().st_size == 0
    with result_file.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"Recorded {len(rows)} measured row(s) in {result_file}")
    return 0


def command_profile(args):
    config, op_root = load_config(args.config)
    runner, torch = load_runner(op_root, config, args.backend)
    profile_config = config["frameworks"][args.backend]["profile"]
    size = args.size if args.size is not None else int(profile_config["size"])
    steps = args.steps if args.steps is not None else int(profile_config["steps"])
    warmup = args.warmup if args.warmup is not None else int(profile_config["warmup"])
    if size < 0 or steps < 1 or warmup < 0:
        raise ValueError("profile size/warmup must be non-negative and steps must be positive")

    values = inputs_for(config["name"], size, torch)
    for _ in range(warmup):
        runner(*values)
    torch.cuda.synchronize()
    configured_output = Path(profile_config["output"])
    output_dir = configured_output if configured_output.is_absolute() else ROOT / configured_output
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir /= f"{Path(config['frameworks'][args.backend]['implementation']).stem}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "trace.json"
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as profiler:
        for _ in range(steps):
            runner(*values)
            profiler.step()
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(trace_path))
    table = profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=40)
    (output_dir / "summary.txt").write_text(table + "\n", encoding="utf-8")
    metadata = {
        "operator": config["name"], "backend": args.backend,
        "implementation": config["frameworks"][args.backend]["implementation"],
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "torch_version": torch.__version__, "triton_version": _triton_version(args.backend),
        "cuda_runtime": str(torch.version.cuda or "unknown"),
        "input_size": size, "warmup": warmup, "profile_steps": steps,
        "trace": trace_path.name,
    }
    import json
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Profiled {config['name']}/{args.backend}; trace and summary: {output_dir}")
    return 0


def _triton_version(backend):
    if backend != "triton":
        return "not_applicable"
    import triton
    return triton.__version__


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("list", "test", "bench", "profile"):
        cmd = sub.add_parser(name)
        cmd.add_argument("config", type=Path, help="operator.yaml path")
        if name not in ("list", "profile"):
            cmd.add_argument("--backend", choices=("pytorch", "triton"), required=True)
        if name == "test":
            cmd.set_defaults(handler=command_test)
        elif name == "bench":
            cmd.add_argument("--sizes", type=lambda v: [int(x) for x in v.split(",")], help="optional comma-separated override; defaults come from operator.yaml")
            cmd.add_argument("--warmup", type=int, help="optional warmup override")
            cmd.add_argument("--iterations", type=int, help="optional measurement iteration override")
            cmd.add_argument("--output", type=Path, help="optional CSV output override")
            cmd.set_defaults(handler=command_bench)
        elif name == "profile":
            cmd.add_argument("--backend", choices=("pytorch", "triton"), required=True)
            cmd.add_argument("--size", type=int, help="optional input-size override; defaults come from operator.yaml")
            cmd.add_argument("--warmup", type=int, help="optional warmup-step override")
            cmd.add_argument("--steps", type=int, help="optional profile-step override")
            cmd.set_defaults(handler=command_profile)
        else:
            cmd.set_defaults(handler=command_list)
    return parser


def main():
    args = make_parser().parse_args()
    try:
        return args.handler(args)
    except (RuntimeError, ValueError, AssertionError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
