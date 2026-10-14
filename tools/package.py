#!/usr/bin/env python3
"""Package one implemented operator backend and its test project under dist/."""

import argparse
import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_FILES = {
    "cuda": [
        "CMakeLists.txt",
        "operator.yaml",
        "docs/spec.md",
        "docs/usage.md",
        "cuda/CMakeLists.txt",
        "cuda/include/ops/add_cuda.hpp",
        "cuda/kernel/add_kernel.cu",
        "cuda/host/add_cuda.cu",
        "tests/cuda/CMakeLists.txt",
        "tests/cuda/correctness/add_cuda_test.cu",
        "tests/cuda/memory/add_cuda_memory_eval.cu",
        "tests/cuda/performance/add_cuda_benchmark.cu",
        "examples/add_demo.cu",
        "tests/cuda/performance/results/cuda.csv",
    ],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", choices=("add",), default="add")
    parser.add_argument("--backend", choices=("cuda", "cann"), required=True)
    parser.add_argument("--output", type=Path, help="output directory (default: dist/<operator>-<version>-<backend>)")
    parser.add_argument("--force", action="store_true", help="replace the selected generated output directory")
    args = parser.parse_args()
    if args.backend not in PACKAGE_FILES:
        print(f"ERROR: {args.operator} {args.backend} has no implemented, packageable backend.", file=sys.stderr)
        return 2

    operator_root = ROOT / "operators" / args.operator
    version = "0.1.0"
    files = PACKAGE_FILES[args.backend]
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
        "files": files,
    }
    (destination / "package.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Packaged {args.operator} backend={args.backend} into {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
