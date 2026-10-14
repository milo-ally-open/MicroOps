#!/usr/bin/env python3
"""Copy one operator backend and its essential metadata into dist/."""

import argparse
import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OPERATOR_ROOT = ROOT / "operators" / "reduce_sum"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", choices=("reduce_sum",), default="reduce_sum")
    parser.add_argument("--backend", choices=("cpu", "cuda", "ascendc"), default="cpu")
    parser.add_argument("--output", type=Path, help="output directory (default: dist/<operator>-<version>-<backend>)")
    parser.add_argument("--force", action="store_true", help="replace the selected generated output directory")
    args = parser.parse_args()

    files_by_backend = {
        "cpu": [
            "reference/include/ops/reduce_sum.hpp",
            "reference/src/reduce_sum.cpp",
        ],
        "cuda": [
            "backends/cuda/CMakeLists.txt",
            "backends/cuda/include/ops/reduce_sum_cuda.hpp",
            "backends/cuda/kernel/reduce_sum_kernel.cuh",
            "backends/cuda/host/reduce_sum_cuda.cu",
        ],
        "ascendc": [
            "backends/ascendc/CMakeLists.txt",
            "backends/ascendc/kernel/reduce_sum.cpp",
            "backends/ascendc/host/reduce_sum_host.cpp",
        ],
    }
    metadata = {
        "operator": args.operator,
        "version": "0.1.0",
        "backend": args.backend,
        "status": {"cpu": "implemented_unverified", "cuda": "scaffold_unverified", "ascendc": "scaffold"}[args.backend],
        "files": ["operator.yaml", "docs/spec.md", "docs/usage.md"] + files_by_backend[args.backend],
    }
    destination = args.output or (ROOT / "dist" / f"{args.operator}-{metadata['version']}-{args.backend}")
    if not destination.is_absolute():
        destination = (Path.cwd() / destination).resolve()

    for relative in metadata["files"]:
        source = OPERATOR_ROOT / relative
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

    for relative in metadata["files"]:
        source = OPERATOR_ROOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (destination / "package.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Packaged {args.operator} backend={args.backend} into {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
