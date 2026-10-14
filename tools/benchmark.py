#!/usr/bin/env python3
"""Run real CPU reference measurements and append environment-tagged CSV rows."""

import argparse
import csv
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIELDS = [
    "timestamp_utc", "backend", "device", "input_size", "dtype", "compiler",
    "compiler_version", "runtime", "iterations", "average_latency_us",
    "throughput_elements_per_second",
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
        print("ERROR: required tool was not found; install CMake 3.21+ and a C++17 compiler.", file=sys.stderr)
        print("       " + str(error), file=sys.stderr)
        return None


def cache_value(cache_path, key):
    prefix = key + ":"
    if not cache_path.exists():
        return ""
    for line in cache_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(prefix):
            return line.split("=", 1)[-1].strip()
    return ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu",), default="cpu",
                        help="only CPU reference benchmarking is implemented")
    parser.add_argument("--sizes", type=positive_ints, default=[1024, 1048576])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "benchmark_results.csv")
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")

    build_dir = ROOT / "build" / "benchmark-cpu"
    build_script = ROOT / "tools" / "build.py"
    built = run([sys.executable, str(build_script), "--backend", "cpu",
                 "--build-dir", str(build_dir), "--benchmarks"])
    if built is None:
        return 127
    if built.returncode != 0:
        return built.returncode

    cache = build_dir / "CMakeCache.txt"
    compiler = cache_value(cache, "CMAKE_CXX_COMPILER") or cache_value(cache, "CMAKE_CXX_COMPILER:FILEPATH")
    if not compiler:
        # CMake cache keys already include the type before '=', so fall back to
        # a direct lookup for common cache declarations.
        for key in ("CMAKE_CXX_COMPILER:FILEPATH", "CMAKE_CXX_COMPILER:STRING"):
            compiler = cache_value(cache, key)
            if compiler:
                break
    compiler = compiler or "unknown (not reported by CMake)"
    version = "unknown"
    if compiler != "unknown (not reported by CMake)":
        version_run = run([compiler, "--version"], capture=True)
        if version_run is not None and version_run.returncode == 0:
            version_lines = (version_run.stdout or version_run.stderr).splitlines()
            if version_lines:
                version = version_lines[0].strip()

    executable_name = "reduce_sum_cpu_benchmark"
    if platform.system() == "Windows":
        executable_name += ".exe"
    executable = build_dir / executable_name
    if not executable.exists():
        executable = build_dir / "Release" / executable_name
    if not executable.exists():
        print(f"ERROR: benchmark executable was not produced: {executable}", file=sys.stderr)
        return 2

    device_name = platform.processor() or platform.uname().processor or platform.machine() or "unknown CPU"
    device = f"CPU: {device_name} ({platform.machine()})"
    runtime = f"C++17 std::chrono::steady_clock; {platform.platform()}"
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for size in args.sizes:
        measured = run([str(executable), str(size), str(args.iterations)], capture=True)
        if measured is None:
            return 127
        if measured.returncode != 0:
            print(measured.stderr, file=sys.stderr)
            return measured.returncode
        try:
            fields = measured.stdout.strip().split(",")
            measured_size, measured_iterations = int(fields[0]), int(fields[1])
            average_us = float(fields[2])
        except (IndexError, ValueError):
            print(f"ERROR: unexpected benchmark output: {measured.stdout!r}", file=sys.stderr)
            return 2
        if measured_size != size or measured_iterations != args.iterations or average_us <= 0:
            print("ERROR: benchmark returned inconsistent or non-positive measurement data.", file=sys.stderr)
            return 2
        rows.append({
            "timestamp_utc": timestamp,
            "backend": "cpu_reference",
            "device": device,
            "input_size": measured_size,
            "dtype": "float32",
            "compiler": compiler,
            "compiler_version": version,
            "runtime": runtime,
            "iterations": measured_iterations,
            "average_latency_us": f"{average_us:.6f}",
            "throughput_elements_per_second": f"{measured_size / (average_us * 1.0e-6):.3f}",
        })

    output = args.output if args.output.is_absolute() else (Path.cwd() / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output.exists() or output.stat().st_size == 0
    with output.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"Recorded {len(rows)} real CPU measurement(s) in {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
