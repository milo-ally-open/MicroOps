#!/usr/bin/env python3
"""Configure, build, and run one operator backend's independent test project."""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command):
    try:
        return subprocess.run(command, cwd=ROOT).returncode
    except FileNotFoundError as error:
        print("ERROR: required build tool was not found. Install CMake 3.21+ and the selected backend toolchain.", file=sys.stderr)
        print("       " + str(error), file=sys.stderr)
        return 127


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", choices=("add",), default="add")
    parser.add_argument("--backend", choices=("cuda", "cann"), required=True)
    parser.add_argument("--build-root", type=Path, default=ROOT / "build",
                        help="test build root (default: build)")
    args = parser.parse_args()
    build_root = args.build_root if args.build_root.is_absolute() else (Path.cwd() / args.build_root).resolve()
    test_project = ROOT / "operators" / args.operator / "tests" / args.backend
    build_dir = build_root / args.operator / "tests" / args.backend

    configure = run(["cmake", "-S", str(test_project), "-B", str(build_dir)])
    if configure != 0:
        print(f"Test project configure failed for operator={args.operator} backend={args.backend}.", file=sys.stderr)
        return configure
    built = run(["cmake", "--build", str(build_dir), "--config", "Release", "--parallel"])
    if built != 0:
        print(f"Test project build failed for operator={args.operator} backend={args.backend}.", file=sys.stderr)
        return built
    tested = run(["ctest", "--test-dir", str(build_dir), "-C", "Release", "--output-on-failure"])
    if tested != 0:
        print(f"Tests failed for operator={args.operator} backend={args.backend}.", file=sys.stderr)
        return tested
    print(f"CTest completed for operator={args.operator} backend={args.backend}; skipped suites are not passes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
