#!/usr/bin/env python3
"""Build and run the registered CUDA performance evaluation for an operator."""

import argparse
import csv
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIELDS = [
    "timestamp_utc", "operator", "backend", "device", "input_size", "dtype",
    "compiler", "compiler_version", "runtime", "iterations", "working_set_bytes",
    "device_memory_total_bytes", "device_memory_free_before_bytes",
    "device_memory_free_after_alloc_bytes", "device_memory_free_after_release_bytes",
    "allocation_status", "average_latency_us", "estimated_bytes_per_iteration",
    "effective_bandwidth_GBps", "validation",
]


def positive_ints(value):
    try:
        values = [int(part.strip()) for part in value.split(",")]
        if not values or any(number <= 0 for number in values):
            raise ValueError
        return values
    except ValueError as error:
        raise argparse.ArgumentTypeError("provide comma-separated positive integers") from error


def run(command, capture=False):
    try:
        return subprocess.run(command, cwd=ROOT, text=True, capture_output=capture)
    except FileNotFoundError as error:
        print("ERROR: required tool was not found. Install CMake 3.21+ and CUDA Toolkit.", file=sys.stderr)
        print("       " + str(error), file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", choices=("add",), default="add")
    parser.add_argument("--sizes", type=positive_ints, default=[1024, 1048576])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path,
                        help="CSV output (default: operators/<op>/benchmarks/cuda/results/cuda.csv)")
    parser.add_argument("--build-dir", type=Path,
                        help="test-project build directory (default: build/<op>/tests/cuda-benchmark)")
    parser.add_argument("--executable", type=Path, help="use an already-built benchmark executable")
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")

    operator_root = ROOT / "operators" / args.operator
    build_dir = args.build_dir or ROOT / "build" / args.operator / "tests" / "cuda-benchmark"
    if not build_dir.is_absolute():
        build_dir = (Path.cwd() / build_dir).resolve()
    executable = args.executable
    if executable is None:
        configured = run(["cmake", "-S", str(operator_root / "tests" / "cuda"), "-B", str(build_dir),
                          "-DOPS_BUILD_PERFORMANCE=ON"])
        if configured is None:
            return 127
        if configured.returncode != 0:
            return configured.returncode
        built = run(["cmake", "--build", str(build_dir), "--config", "Release",
                     "--target", "add_cuda_benchmark", "--parallel"])
        if built is None:
            return 127
        if built.returncode != 0:
            return built.returncode
        executable = build_dir / "add_cuda_benchmark"
        if platform.system() == "Windows":
            executable = executable.with_suffix(".exe")
        if not executable.exists():
            executable = build_dir / "Release" / executable.name
    elif not executable.is_absolute():
        executable = (Path.cwd() / executable).resolve()
    if not executable.is_file():
        print(f"ERROR: CUDA benchmark executable was not found: {executable}", file=sys.stderr)
        return 2

    nvcc = shutil.which("nvcc")
    compiler_version = "unavailable"
    if nvcc:
        version = run([nvcc, "--version"], capture=True)
        if version is not None and version.returncode == 0:
            matches = [line.strip() for line in (version.stdout or version.stderr).splitlines()
                       if "release" in line.lower()]
            if matches:
                compiler_version = matches[-1]

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for size in args.sizes:
        if size > 2147483647:
            print(f"ERROR: CUDA add v1 accepts N <= INT_MAX; got {size}.", file=sys.stderr)
            return 2
        measured = run([str(executable), str(size), str(args.iterations)], capture=True)
        if measured is None:
            return 127
        fields = measured.stdout.strip().split("\t")
        if len(fields) != 14 or fields[0] != "CUDA_RESULT":
            print(measured.stderr or f"ERROR: unexpected CUDA benchmark output: {measured.stdout!r}", file=sys.stderr)
            return measured.returncode or 2
        try:
            device, runtime_version, driver_version = fields[1:4]
            measured_size, working_set = int(fields[4]), int(fields[5])
            memory_free_before, memory_total = int(fields[6]), int(fields[7])
            memory_free_after_alloc, memory_free_after_release = int(fields[8]), int(fields[9])
            allocation_status = fields[10]
            average_latency, bandwidth = float(fields[11]), float(fields[12])
            validation = fields[13]
        except ValueError:
            print(f"ERROR: malformed CUDA benchmark output: {measured.stdout!r}", file=sys.stderr)
            return 2
        if measured_size != size:
            print("ERROR: CUDA benchmark returned an unexpected input size.", file=sys.stderr)
            return 2
        allocation_ok = allocation_status == "allocated"
        if allocation_ok and (measured.returncode != 0 or average_latency <= 0 or bandwidth <= 0 or validation != "passed"):
            print(f"ERROR: CUDA evaluation failed for N={size}: {validation}", file=sys.stderr)
            return measured.returncode or 1
        rows.append({
            "timestamp_utc": timestamp, "operator": args.operator, "backend": "cuda",
            "device": device, "input_size": measured_size, "dtype": "float32",
            "compiler": "nvcc", "compiler_version": compiler_version,
            "runtime": f"CUDA Runtime {runtime_version}; driver API {driver_version}",
            "iterations": args.iterations, "working_set_bytes": working_set,
            "device_memory_total_bytes": memory_total,
            "device_memory_free_before_bytes": memory_free_before,
            "device_memory_free_after_alloc_bytes": memory_free_after_alloc,
            "device_memory_free_after_release_bytes": memory_free_after_release,
            "allocation_status": allocation_status,
            "average_latency_us": f"{average_latency:.6f}" if allocation_ok else "",
            "estimated_bytes_per_iteration": working_set,
            "effective_bandwidth_GBps": f"{bandwidth:.6f}" if allocation_ok else "",
            "validation": validation,
        })
        if not allocation_ok:
            print(f"WARNING: N={size} allocation did not fit: {allocation_status}", file=sys.stderr)
        elif measured.returncode != 0:
            print(measured.stderr or f"CUDA benchmark exited {measured.returncode}", file=sys.stderr)
            return measured.returncode

    output = args.output or (operator_root / "benchmarks" / "cuda" / "results" / "cuda.csv")
    if not output.is_absolute():
        output = (Path.cwd() / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output.exists() or output.stat().st_size == 0
    with output.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"Recorded {len(rows)} real CUDA evaluation row(s) in {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
