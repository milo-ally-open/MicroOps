#!/usr/bin/env python3
"""Configure and build one selected operator device backend."""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command):
    try:
        return subprocess.run(command, cwd=ROOT).returncode
    except FileNotFoundError as error:
        print("ERROR: CMake was not found. Install CMake 3.21+ and the selected backend toolchain.", file=sys.stderr)
        print("       " + str(error), file=sys.stderr)
        return 127


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", choices=("add",), default="add")
    parser.add_argument("--backend", choices=("cuda", "cann"), required=True)
    parser.add_argument("--build-dir", type=Path,
                        help="build directory (default: build/<operator>/<backend>)")
    parser.add_argument("--examples", action="store_true", help="also build operator examples")
    args = parser.parse_args()

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
    result = run(command)
    if result != 0:
        return result
    return run(["cmake", "--build", str(build_dir), "--config", "Release", "--parallel"])


if __name__ == "__main__":
    sys.exit(main())
