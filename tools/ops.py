#!/usr/bin/env python3
"""Unified build, test, performance-evaluation, and packaging entry point."""

import argparse
import csv
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OPERATORS = tuple(sorted(path.name for path in (ROOT / "operators").iterdir()
                         if path.is_dir() and (path / "operator.yaml").is_file()))
CUDA_PACKAGE_FILES = (
    "CMakeLists.txt",
    "operator.yaml",
    "docs/spec.md",
    "docs/usage.md",
    "src/cuda/CMakeLists.txt",
    "src/cuda/include/ops/add_cuda.hpp",
    "src/cuda/kernel/add_kernel.cu",
    "src/cuda/host/add_cuda.cu",
    "tests/cuda/CMakeLists.txt",
    "tests/cuda/correctness/add_cuda_test.cu",
    "tests/cuda/memory/add_cuda_memory_eval.cu",
    "tests/cuda/performance/add_cuda_benchmark.cu",
    "examples/cuda/add_demo.cu",
    "tests/cuda/performance/results/cuda.csv",
)
CSV_FIELDS = (
    "timestamp_utc", "operator", "backend", "device", "input_size", "dtype",
    "compiler", "compiler_version", "runtime", "iterations", "working_set_bytes",
    "device_memory_total_bytes", "device_memory_free_before_bytes",
    "device_memory_free_after_alloc_bytes", "device_memory_free_after_release_bytes",
    "allocation_status", "average_latency_us", "estimated_bytes_per_iteration",
    "effective_bandwidth_GBps", "validation",
)


def run(command, *, capture=False, missing_tool_message="ERROR: required tool was not found."):
    try:
        return subprocess.run(command, cwd=ROOT, text=True, capture_output=capture)
    except FileNotFoundError as error:
        print(missing_tool_message, file=sys.stderr)
        print(f"       {error}", file=sys.stderr)
        return None


def add_operator_argument(parser):
    parser.add_argument("--operator", choices=OPERATORS, default="add")


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="configure and build one backend")
    add_operator_argument(build)
    build.add_argument("--backend", choices=("cuda", "cann"), required=True)
    build.add_argument("--build-dir", type=Path,
                       help="build directory (default: build/<operator>/<backend>)")
    build.add_argument("--examples", action="store_true", help="also build operator examples")
    build.set_defaults(handler=build_backend)

    test = commands.add_parser("test", help="build and run one backend test project")
    add_operator_argument(test)
    test.add_argument("--backend", choices=("cuda", "cann"), required=True)
    test.add_argument("--build-root", type=Path, default=ROOT / "build",
                      help="test build root (default: build)")
    test.set_defaults(handler=test_backend)

    benchmark = commands.add_parser("benchmark", help="run a registered performance evaluation")
    add_operator_argument(benchmark)
    benchmark.add_argument("--backend", choices=("cuda",), default="cuda")
    benchmark.add_argument("--sizes", type=positive_ints, default=[1024, 1048576])
    benchmark.add_argument("--iterations", type=int, default=100)
    benchmark.add_argument("--output", type=Path,
                           help="CSV output (default: operators/<op>/tests/cuda/performance/results/cuda.csv)")
    benchmark.add_argument("--build-dir", type=Path,
                           help="test build directory (default: build/<op>/tests/cuda-benchmark)")
    benchmark.add_argument("--executable", type=Path, help="use an already-built evaluation executable")
    benchmark.set_defaults(handler=run_benchmark)

    package = commands.add_parser("package", help="package an implemented backend under dist/")
    add_operator_argument(package)
    package.add_argument("--backend", choices=("cuda", "cann"), required=True)
    package.add_argument("--output", type=Path,
                         help="output directory (default: dist/<operator>-<version>-<backend>)")
    package.add_argument("--force", action="store_true",
                         help="replace the selected generated output directory")
    package.set_defaults(handler=package_backend)
    return parser


def build_backend(args):
    operator_root = ROOT / "operators" / args.operator
    build_dir = args.build_dir or ROOT / "build" / args.operator / args.backend
    if not build_dir.is_absolute():
        build_dir = (Path.cwd() / build_dir).resolve()
    command = [
        "cmake", "-S", str(operator_root), "-B", str(build_dir),
        f"-DOPS_ENABLE_CUDA={'ON' if args.backend == 'cuda' else 'OFF'}",
        f"-DOPS_ENABLE_CANN={'ON' if args.backend == 'cann' else 'OFF'}",
        f"-DOPS_BUILD_EXAMPLES={'ON' if args.examples else 'OFF'}",
    ]
    print(f"Configuring operator={args.operator} backend={args.backend} in {build_dir}")
    configured = run(command, missing_tool_message=
                     "ERROR: CMake was not found. Install CMake 3.21+ and the selected backend toolchain.")
    if configured is None:
        return 127
    if configured.returncode != 0:
        return configured.returncode
    built = run(["cmake", "--build", str(build_dir), "--config", "Release", "--parallel"],
                missing_tool_message="ERROR: CMake build tool was not found.")
    return 127 if built is None else built.returncode


def test_backend(args):
    build_root = args.build_root if args.build_root.is_absolute() else (Path.cwd() / args.build_root).resolve()
    test_project = ROOT / "operators" / args.operator / "tests" / args.backend
    build_dir = build_root / args.operator / "tests" / args.backend
    configured = run(["cmake", "-S", str(test_project), "-B", str(build_dir)],
                     missing_tool_message=
                     "ERROR: required build tool was not found. Install CMake 3.21+ and the selected backend toolchain.")
    if configured is None:
        return 127
    if configured.returncode != 0:
        print(f"Test project configure failed for operator={args.operator} backend={args.backend}.", file=sys.stderr)
        return configured.returncode
    built = run(["cmake", "--build", str(build_dir), "--config", "Release", "--parallel"])
    if built is None:
        return 127
    if built.returncode != 0:
        print(f"Test project build failed for operator={args.operator} backend={args.backend}.", file=sys.stderr)
        return built.returncode
    tested = run(["ctest", "--test-dir", str(build_dir), "-C", "Release", "--output-on-failure"],
                 missing_tool_message="ERROR: CTest was not found; install CMake with CTest support.")
    if tested is None:
        return 127
    if tested.returncode != 0:
        print(f"Tests failed for operator={args.operator} backend={args.backend}.", file=sys.stderr)
        return tested.returncode
    print(f"CTest completed for operator={args.operator} backend={args.backend}; skipped suites are not passes.")
    return 0


def positive_ints(value):
    try:
        values = [int(part.strip()) for part in value.split(",")]
        if not values or any(number <= 0 for number in values):
            raise ValueError
        return values
    except ValueError as error:
        raise argparse.ArgumentTypeError("provide comma-separated positive integers") from error


def run_benchmark(args):
    if args.iterations <= 0:
        print("ERROR: --iterations must be positive.", file=sys.stderr)
        return 2
    operator_root = ROOT / "operators" / args.operator
    build_dir = args.build_dir or ROOT / "build" / args.operator / "tests" / "cuda-benchmark"
    if not build_dir.is_absolute():
        build_dir = (Path.cwd() / build_dir).resolve()
    executable = args.executable
    if executable is None:
        configured = run(["cmake", "-S", str(operator_root / "tests" / "cuda"), "-B", str(build_dir),
                          "-DOPS_BUILD_PERFORMANCE=ON"], missing_tool_message=
                         "ERROR: required tool was not found. Install CMake 3.21+ and CUDA Toolkit.")
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
        print(f"ERROR: CUDA evaluation executable was not found: {executable}", file=sys.stderr)
        return 2

    compiler_version = "unavailable"
    nvcc = shutil.which("nvcc")
    if nvcc:
        version = run([nvcc, "--version"], capture=True)
        if version is not None and version.returncode == 0:
            release_lines = [line.strip() for line in (version.stdout or version.stderr).splitlines()
                             if "release" in line.lower()]
            if release_lines:
                compiler_version = release_lines[-1]

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
            print(measured.stderr or f"ERROR: unexpected evaluation output: {measured.stdout!r}", file=sys.stderr)
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
            print(f"ERROR: malformed CUDA evaluation output: {measured.stdout!r}", file=sys.stderr)
            return 2
        if measured_size != size:
            print("ERROR: CUDA evaluation returned an unexpected input size.", file=sys.stderr)
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
            print(measured.stderr or f"CUDA evaluation exited {measured.returncode}", file=sys.stderr)
            return measured.returncode

    output = args.output or (operator_root / "tests" / "cuda" / "performance" / "results" / "cuda.csv")
    if not output.is_absolute():
        output = (Path.cwd() / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output.exists() or output.stat().st_size == 0
    with output.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"Recorded {len(rows)} real CUDA evaluation row(s) in {output}")
    return 0


def package_backend(args):
    if args.backend != "cuda" or args.operator != "add":
        print(f"ERROR: {args.operator} {args.backend} has no implemented, packageable backend.", file=sys.stderr)
        return 2
    operator_root = ROOT / "operators" / args.operator
    version = "0.1.0"
    files = CUDA_PACKAGE_FILES
    destination = args.output or ROOT / "dist" / f"{args.operator}-{version}-{args.backend}"
    if not destination.is_absolute():
        destination = (Path.cwd() / destination).resolve()
    for relative in files:
        source = operator_root / relative
        if not source.is_file():
            print(f"ERROR: required package file is missing: {source}", file=sys.stderr)
            return 2

    if destination.exists():
        if not args.force:
            print(f"ERROR: {destination} already exists; pass --force to replace that generated package.", file=sys.stderr)
            return 2
        old_manifest = destination / "package.json"
        try:
            previous_manifest = json.loads(old_manifest.read_text(encoding="utf-8"))
            previous_files = previous_manifest["files"]
            if previous_manifest.get("operator") != args.operator:
                raise ValueError("package belongs to a different operator")
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"ERROR: refusing to replace an unrecognized directory: {destination} ({error})", file=sys.stderr)
            return 2
        for relative in previous_files + ["package.json"]:
            target = (destination / relative).resolve()
            if not target.is_relative_to(destination.resolve()):
                print(f"ERROR: package manifest contains an unsafe path: {relative}", file=sys.stderr)
                return 2
            if target.is_file() or target.is_symlink():
                target.unlink()

    for relative in files:
        source = operator_root / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    manifest = {
        "operator": args.operator,
        "version": version,
        "backend": args.backend,
        "status": "device_tested_cmake_unverified",
        "files": list(files),
    }
    (destination / "package.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Packaged {args.operator} backend={args.backend} into {destination}")
    return 0


def main():
    args = make_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
