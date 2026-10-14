#!/usr/bin/env python3
"""Build, test, benchmark, and package an operator's CUDA implementation."""

from operator_common import main


CUDA_PROFILE = {
    "name": "CUDA",
    "benchmark_protocol": "cuda_tsv_v1",
    "result_tag": "CUDA_RESULT",
    "compiler": "nvcc",
    "runtime_format": "CUDA Runtime {runtime_version}; driver API {driver_version}",
    "environment_hint": "Install/activate the CUDA Toolkit or add nvcc to PATH.",
}


if __name__ == "__main__":
    raise SystemExit(main("cuda", CUDA_PROFILE))
