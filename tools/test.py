#!/usr/bin/env python3
"""Build and run CPU reference tests, with explicit optional-backend status."""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command):
    try:
        return subprocess.run(command, cwd=ROOT).returncode
    except FileNotFoundError as error:
        print("ERROR: required build tool was not found. Install CMake 3.21+ and a C++17 compiler.", file=sys.stderr)
        print("       " + str(error), file=sys.stderr)
        return 127


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda", "ascendc"), default="cpu")
    parser.add_argument("--build-root", type=Path, default=ROOT / "build")
    args = parser.parse_args()
    build_root = args.build_root if args.build_root.is_absolute() else (Path.cwd() / args.build_root).resolve()

    cpu_build = build_root / "cpu"
    build_script = ROOT / "tools" / "build.py"
    result = run([sys.executable, str(build_script), "--backend", "cpu", "--build-dir", str(cpu_build)])
    if result != 0:
        return result
    result = run(["ctest", "--test-dir", str(cpu_build), "-C", "Release", "--output-on-failure"])
    if result != 0:
        print("CPU reference tests failed.", file=sys.stderr)
        return result
    print("CPU reference tests: PASS")

    if args.backend == "cpu":
        return 0

    backend_build = build_root / args.backend
    result = run([sys.executable, str(build_script), "--backend", args.backend, "--build-dir", str(backend_build)])
    if result != 0:
        print(f"{args.backend} backend configure/build: FAILED or unavailable (see the diagnostic above).", file=sys.stderr)
        print("CPU reference test result remains PASS; no device test is counted as passed.", file=sys.stderr)
        return result

    if args.backend == "ascendc":
        # Its CMake scaffold currently exits with a diagnostic, so reaching this
        # branch would indicate that the scaffold has been replaced.
        print("Ascend C backend built, but device correctness tests are not registered: SKIP")
    else:
        print("CUDA backend built, but device correctness tests are not registered: SKIP")
    return 0


if __name__ == "__main__":
    sys.exit(main())
