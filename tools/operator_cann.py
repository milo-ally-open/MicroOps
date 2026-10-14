#!/usr/bin/env python3
"""Build, test, benchmark, and package an operator's CANN implementation."""

from operator_common import main


CANN_PROFILE = {
    "name": "CANN",
    "benchmark_protocol": "cann_tsv_v1",
    "result_tag": "CANN_RESULT",
    "compiler": "bisheng",
    "runtime_format": "CANN runtime {runtime_version}; driver version {driver_version}",
    "environment_hint": "Source the matching CANN set_env.sh so bisheng is available on PATH.",
}


if __name__ == "__main__":
    raise SystemExit(main("cann", CANN_PROFILE))
