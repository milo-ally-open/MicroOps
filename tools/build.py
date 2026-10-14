#!/usr/bin/env python3
"""Configure and build one independently selected OpsFromScratch backend."""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command):
    try:
        return subprocess.run(command, cwd=ROOT).returncode
    except FileNotFoundError as error:
        print("ERROR: CMake was not found. Install CMake 3.21+ and a C++17 compiler.", file=sys.stderr)
        print("       " + str(error), file=sys.stderr)
        return 127


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda", "ascendc"), default="cpu")
    parser.add_argument("--build-dir", type=Path, help="build directory (default: build/<backend>)")
    parser.add_argument("--examples", action="store_true", help="also build operator examples")
    parser.add_argument("--benchmarks", action="store_true", help="also build benchmark executables")
    parser.add_argument("--no-tests", action="store_true", help="do not configure CPU tests")
    args = parser.parse_args()

    build_dir = args.build_dir or ROOT / "build" / args.backend
    if not build_dir.is_absolute():
        build_dir = (Path.cwd() / build_dir).resolve()
    cuda = "ON" if args.backend == "cuda" else "OFF"
    ascendc = "ON" if args.backend == "ascendc" else "OFF"
    command = [
        "cmake", "-S", str(ROOT), "-B", str(build_dir),
        f"-DOPS_ENABLE_CUDA={cuda}",
        f"-DOPS_ENABLE_ASCENDC={ascendc}",
        f"-DOPS_BUILD_TESTS={'OFF' if args.no_tests else 'ON'}",
        f"-DOPS_BUILD_EXAMPLES={'ON' if args.examples else 'OFF'}",
        f"-DOPS_BUILD_BENCHMARKS={'ON' if args.benchmarks else 'OFF'}",
    ]
    print(f"Configuring backend={args.backend} in {build_dir}")
    result = run(command)
    if result != 0:
        if args.backend == "ascendc":
            print("Ascend C was not built: its kernel/host are still scaffold files. See operators/reduce_sum/docs/tasks.md.", file=sys.stderr)
        return result
    return run(["cmake", "--build", str(build_dir), "--config", "Release", "--parallel"])


if __name__ == "__main__":
    sys.exit(main())
