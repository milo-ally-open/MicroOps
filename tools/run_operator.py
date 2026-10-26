#!/usr/bin/env python3
"""Unified test, bench, and profile entry point for all operator backends."""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
BACKENDS = ("pytorch", "triton", "cuda", "cann")


def positive_csv(value: str) -> str:
    try:
        numbers = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers") from error
    if not numbers or any(number <= 0 for number in numbers):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return ",".join(str(number) for number in numbers)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    for action in ("test", "bench", "profile"):
        command = actions.add_parser(action, help=f"run {action} for one operator backend")
        command.add_argument("config", type=Path, help="operator.yaml path")
        command.add_argument("--backend", choices=BACKENDS, required=True)
        command.add_argument("--kernel", help="select a CUDA/CANN kernel source from operator.yaml")
        command.add_argument("--host", help="select a CUDA/CANN host source from operator.yaml")
        if action == "test":
            command.add_argument("--build-dir", type=Path)
        elif action == "bench":
            command.add_argument("--sizes", type=positive_csv, help="optional comma-separated size override")
            command.add_argument("--iterations", type=int, help="optional iteration override")
            command.add_argument("--output", type=Path)
            command.add_argument("--build-dir", type=Path)
            command.add_argument("--warmup", type=int)
        else:
            command.add_argument("--size", type=int, help="profile input size; defaults to operator.yaml")
            command.add_argument("--iterations", type=int, help="profile iterations; defaults to operator.yaml")
            command.add_argument("--build-dir", type=Path)
    return parser


def run_framework(args) -> int:
    if args.kernel or args.host:
        raise RuntimeError("--kernel/--host selection applies to CUDA/CANN; PyTorch/Triton source comes from operator.yaml")
    if args.build_dir:
        raise RuntimeError("--build-dir applies to native CUDA/CANN test and bench builds")
    command = [sys.executable, str(TOOLS / "operator_framework.py"), args.action,
               str(args.config), "--backend", args.backend]
    if args.action == "bench":
        for name in ("sizes", "iterations", "warmup", "output"):
            value = getattr(args, name)
            if value is not None:
                command.extend((f"--{name}", str(value)))
    elif args.action == "profile":
        for name in ("size", "iterations"):
            value = getattr(args, name)
            if value is not None:
                command.extend((f"--{name if name != 'iterations' else 'steps'}", str(value)))
    return subprocess.run(command, cwd=ROOT).returncode


def run_native_backend_action(args) -> int:
    import operator_common

    backend_profiles = {
        "cuda": {
            "name": "CUDA", "compiler": "nvcc",
            "environment_hint": "Install/activate the CUDA Toolkit or add nvcc to PATH.",
            "benchmark_protocol": "cuda_tsv_v1",
            "result_tag": "CUDA_RESULT",
            "runtime_format": "CUDA Runtime {runtime_version}; driver API {driver_version}",
        },
        "cann": {
            "name": "CANN", "compiler": "bisheng",
            "environment_hint": "Source the matching CANN set_env.sh so bisheng is available on PATH.",
            "benchmark_protocol": "cann_tsv_v1",
            "result_tag": "CANN_RESULT",
            "runtime_format": "CANN runtime {runtime_version}; driver version {driver_version}",
        },
    }
    profile = backend_profiles[args.backend]
    if operator_common.check_backend_environment(profile):
        return 2
    common_args = SimpleNamespace(
        config=args.config, backend=args.backend, backend_profile=profile,
        kernel=args.kernel, host=args.host, build_dir=args.build_dir,
    )
    if args.action == "test":
        return operator_common.test_backend(common_args)
    common_args.sizes = [int(item) for item in args.sizes.split(",")] if args.sizes else None
    common_args.iterations = args.iterations
    common_args.output = args.output
    common_args.executable = None
    if args.warmup is not None:
        print("NOTE: CUDA/CANN bench warmup policy is defined by the test harness.")
    return operator_common.run_bench(common_args)


def backend_profile(args):
    import operator_common

    context = operator_common.resolve_selection(SimpleNamespace(
        config=args.config, backend=args.backend, kernel=args.kernel, host=args.host))
    if context is None:
        raise RuntimeError("could not resolve the selected operator source files")
    config = context["backend"].get("profile")
    if not config:
        raise RuntimeError(f"{args.backend} profile is not configured in {context['config_path']}")
    build_context = {
        "name": args.backend.upper(),
        "compiler": "nvcc" if args.backend == "cuda" else "bisheng",
        "environment_hint": (
            "Install Nsight Compute and expose ncu on PATH." if args.backend == "cuda"
            else "Source the matching CANN set_env.sh so bisheng and msprof are on PATH."
        ),
    }
    status = operator_common.check_backend_environment(build_context)
    if status:
        raise RuntimeError(f"{args.backend} compiler environment check failed")
    tool_name = config["tool"]
    tool_path = shutil.which(tool_name)
    profile_mode = "hardware-counters" if args.backend == "cuda" else "device-timeline"
    if not tool_path and args.backend == "cuda":
        fallback = config.get("fallback_tool", "nsys")
        tool_path = shutil.which(fallback)
        if tool_path:
            tool_name = fallback
            profile_mode = "cuda-timeline"
    if not tool_path:
        raise RuntimeError(f"{tool_name} is required for {args.backend} profiling; {build_context['environment_hint']}")
    tests = context["backend"].get("tests", {})
    bench = context["backend"].get("bench", {})
    if not tests.get("project") or not bench.get("target") or not bench.get("executable"):
        raise RuntimeError(f"{args.backend} needs tests.project and bench target/executable in operator.yaml")

    build_dir = (operator_common.repository_path(args.build_dir) if args.build_dir
                 else operator_common.default_build_dir(context, bench=True))
    options = dict(tests.get("options", {}))
    options.update(bench.get("options", {}))
    source_dir = operator_common.get_cmake_dir(context["project_root"], tests["project"])
    configure = ["cmake", "-S", str(source_dir), "-B", str(build_dir),
                 *operator_common.source_definitions(context, {"options": options}, include_tests=True)]
    result = operator_common.run(configure)
    if result is None or result.returncode:
        return 127 if result is None else result.returncode
    result = operator_common.run(["cmake", "--build", str(build_dir), "--config", "Release",
                                  "--target", bench["target"], "--parallel"])
    if result is None or result.returncode:
        return 127 if result is None else result.returncode
    executable = (build_dir / bench["executable"]).resolve()
    if not executable.is_file():
        raise RuntimeError(f"profile executable was not produced: {executable}")

    size = args.size if args.size is not None else int(config["size"])
    iterations = args.iterations if args.iterations is not None else int(config["iterations"])
    if size <= 0 or iterations <= 0:
        raise RuntimeError("profile size and iterations must be positive")
    output_root = operator_common.repository_path(config["output"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_root / context["selection"] / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.backend == "cuda" and tool_name == "ncu":
        report = output_dir / "kernel_profile"
        command = [tool_path, "--set", "basic", "--launch-skip", str(config["launch_skip"]),
                   "--launch-count", str(config["launch_count"]), "--export", str(report),
                   str(executable), str(size), str(iterations)]
    elif args.backend == "cuda":
        report = output_dir / "cuda_timeline"
        command = [tool_path, "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none",
                   "--stats=true", "--force-overwrite=true", "--output", str(report),
                   str(executable), str(size), str(iterations)]
    else:
        application = shlex.join([str(executable), str(size), str(iterations)])
        command = [tool_path, f"--output={output_dir}", f"--application={application}"]
    print(f"Profiling {context['config']['name']} backend={args.backend} mode={profile_mode} "
          f"kernel={context['kernel_file']} host={context['host_file']} N={size}")
    if profile_mode == "cuda-timeline":
        print("NOTE: ncu is unavailable; using Nsight Systems for CUDA API/kernel timeline. "
              "This does not collect Nsight Compute hardware counters.")
    result = operator_common.run(command, capture=(args.backend == "cuda" and tool_name == "ncu"))
    if result is None:
        return 127
    if args.backend == "cuda" and tool_name == "ncu":
        captured_output = (result.stdout or "") + (result.stderr or "")
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        if result.returncode and "ERR_NVGPUCTRPERM" in captured_output:
            nsys = shutil.which("nsys")
            if nsys:
                print("NOTE: NCU hardware counters are blocked by driver permissions; "
                      "falling back to Nsight Systems CUDA timeline (no hardware counters).")
                report = output_dir / "cuda_timeline"
                command = [nsys, "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none",
                           "--stats=true", "--force-overwrite=true", "--output", str(report),
                           str(executable), str(size), str(iterations)]
                profile_mode = "cuda-timeline"
                result = operator_common.run(command)
                if result is None:
                    return 127
    if result.returncode:
        return result.returncode
    print(f"Profile artifacts: {output_dir}")
    return 0


def check_declared_status(args) -> int:
    """Skip scaffold implementations before probing a local toolchain."""
    if args.backend in ("pytorch", "triton"):
        import operator_framework

        config, _ = operator_framework.load_config(args.config)
        profile = config.get("frameworks", {}).get(args.backend)
    else:
        import operator_common

        data = operator_common.load_config_data(args.config)
        if data is None:
            return 2
        profile = data["config"].get("backends", {}).get(args.backend)

    if profile is None:
        print(f"ERROR: {args.backend} is not configured for this operator.", file=sys.stderr)
        return 2
    status = str(profile.get("status", "unknown")).lower()
    if status in {"scaffold", "not_implemented", "unavailable"}:
        print(f"SKIP: {args.config} backend={args.backend} status={status}; "
              "no test, benchmark, or profile was run and no result was recorded.")
        return 77
    return 0


def main() -> int:
    args = make_parser().parse_args()
    try:
        status = check_declared_status(args)
        if status:
            return status
        if args.backend in ("pytorch", "triton"):
            if args.action == "profile" and args.iterations is not None:
                # operator_framework calls profile iterations "steps".
                args.steps = args.iterations
            return run_framework(args)
        if args.action == "profile":
            return backend_profile(args)
        return run_native_backend_action(args)
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
