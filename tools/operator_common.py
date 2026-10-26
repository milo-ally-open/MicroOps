#!/usr/bin/env python3
"""Shared build, package, test, and bench helpers for native backends."""

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
SOURCE_EXTENSIONS = {"cuda": (".cu",), "cann": (".cpp", ".cc", ".cxx")}
CSV_FIELDS = (
    "timestamp_utc", "operator", "backend", "kernel_file", "host_file", "device", "input_size", "dtype",
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


def check_backend_environment(backend_profile):
    compiler = backend_profile["compiler"]
    compiler_path = shutil.which(compiler)
    if compiler_path is None:
        print(f"ERROR: {backend_profile['name']} environment was not detected: "
              f"required compiler {compiler!r} is not on PATH.", file=sys.stderr)
        print(f"       {backend_profile['environment_hint']}", file=sys.stderr)
        return 2
    try:
        result = subprocess.run([compiler_path, "--version"], cwd=ROOT, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"ERROR: could not verify {backend_profile['name']} compiler {compiler!r}: {error}",
              file=sys.stderr)
        return 2
    if result.returncode != 0:
        print(f"ERROR: {backend_profile['name']} compiler {compiler!r} failed its version check.",
              file=sys.stderr)
        if result.stdout:
            print(result.stdout.rstrip(), file=sys.stderr)
        return 2
    return 0


def add_config_argument(parser):
    parser.add_argument("config", type=Path, help="operator.yaml for the selected operator")


def add_selection_arguments(parser):
    add_config_argument(parser)
    parser.add_argument("--kernel", help="override the kernel filename selected in operator.yaml")
    parser.add_argument("--host", help="override the host filename selected in operator.yaml")


def make_parser(backend_name, backend_profile):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    listing = commands.add_parser("list", help=f"list {backend_name.upper()} source files")
    add_config_argument(listing)
    listing.set_defaults(backend=backend_name, backend_profile=backend_profile)
    listing.set_defaults(handler=list_sources)

    build = commands.add_parser("build", help=f"configure and build the {backend_name.upper()} backend")
    add_selection_arguments(build)
    build.set_defaults(backend=backend_name, backend_profile=backend_profile)
    build.add_argument("--build-dir", type=Path, help="override the configured build directory")
    build.add_argument("--examples", action="store_true", help="also build configured examples")
    build.set_defaults(handler=build_backend)

    package = commands.add_parser("package", help=f"package {backend_name.upper()} files under dist/")
    add_selection_arguments(package)
    package.set_defaults(backend=backend_name, backend_profile=backend_profile)
    package.add_argument("--output", type=Path, help="override the configured package output directory")
    package.add_argument("--force", action="store_true", help="replace a recognized generated package")
    package.set_defaults(handler=package_backend)
    return parser


def load_config_data(config_argument):
    config_path = config_argument if config_argument.is_absolute() else ROOT / config_argument
    config_path = config_path.resolve()
    if not config_path.is_file():
        print(f"ERROR: operator configuration was not found: {config_path}", file=sys.stderr)
        return None
    try:
        import yaml
    except ImportError:
        print("ERROR: reading operator.yaml requires PyYAML. Install it with: python3 -m pip install PyYAML",
              file=sys.stderr)
        return None
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        print(f"ERROR: could not read operator configuration {config_path}: {error}", file=sys.stderr)
        return None
    if not isinstance(config, dict) or not config.get("name"):
        print(f"ERROR: {config_path} must contain a top-level operator name.", file=sys.stderr)
        return None
    project_root = (config_path.parent / config.get("root", ".")).resolve()
    return {"config": config, "config_path": config_path, "project_root": project_root}


def load_context(config_argument, backend_override=None):
    data = load_config_data(config_argument)
    if data is None:
        return None
    config = data["config"]
    backend_name = backend_override or config.get("default_backend")
    backends = config.get("backends")
    if not isinstance(backends, dict) or not backends:
        print(f"ERROR: {data['config_path']} has no backend configuration.", file=sys.stderr)
        return None
    if backend_name not in backends:
        print(f"ERROR: backend {backend_name!r} is not configured in {data['config_path']}. "
              f"Available: {', '.join(backends)}", file=sys.stderr)
        return None
    return {
        **data,
        "backend_name": backend_name,
        "backend": backends[backend_name],
    }


def resolve_path(project_root, value):
    path = (project_root / value).resolve()
    if not path.is_relative_to(project_root.resolve()):
        raise ValueError(f"configured path escapes the operator root: {value}")
    return path


def get_cmake_dir(project_root, value):
    path = resolve_path(project_root, value)
    if path.is_file() and path.name == "CMakeLists.txt":
        return path.parent
    return path


def configured_sources(context, role):
    backend = context["backend"]
    source_config = backend.get(role, {})
    directory = resolve_path(context["project_root"], source_config.get("dir", ""))
    extensions = tuple(source_config.get("extensions", SOURCE_EXTENSIONS.get(context["backend_name"], ())))
    if not directory.is_dir():
        return []
    return sorted((path for path in directory.iterdir()
                   if path.is_file() and path.suffix.lower() in extensions), key=lambda path: path.name)


def list_sources(args):
    data = load_config_data(args.config)
    if data is None:
        return 2
    backends = data["config"].get("backends", {})
    if not isinstance(backends, dict) or not backends:
        print(f"ERROR: no backends configured in {data['config_path']}.", file=sys.stderr)
        return 2
    backend_names = (args.backend,) if args.backend else tuple(backends)
    for backend_name in backend_names:
        if backend_name not in backends:
            print(f"ERROR: backend {backend_name!r} is not configured. Available: {', '.join(backends)}",
                  file=sys.stderr)
            return 2
        context = {**data, "backend_name": backend_name, "backend": backends[backend_name]}
        list_sources_for_backend(context, backend_name)
    return 0


def list_sources_for_backend(config_context, backend_name):
    context = config_context
    for role in ("kernel", "host"):
        sources = configured_sources(context, role)
        source_dir = context["backend"].get(role, {}).get("dir", "")
        if sources:
            for source in sources:
                print(f"{context['config']['name']} {backend_name} {role}: {source_dir}/{source.name}")
        else:
            print(f"{context['config']['name']} {backend_name} {role}: no source files (scaffold or not implemented)")
    return 0


def select_source(context, role, override):
    source_config = context["backend"].get(role, {})
    sources = configured_sources(context, role)
    names = [path.name for path in sources]
    requested = override or source_config.get("default_file")
    if requested:
        if Path(requested).name != requested or "/" in requested or "\\" in requested or requested not in names:
            print(f"ERROR: {role} file {requested!r} is not present in configured directory "
                  f"{source_config.get('dir')!r}. Available: {', '.join(names) or '(none)'}", file=sys.stderr)
            return None
        return requested
    if len(names) == 1:
        return names[0]
    if not names:
        print(f"ERROR: no {role} source files in {source_config.get('dir')!r} for "
              f"{context['backend_name']}.", file=sys.stderr)
    else:
        print(f"ERROR: multiple {role} files are available; set {role}.default_file in operator.yaml "
              f"or pass --{role}. Available: {', '.join(names)}", file=sys.stderr)
    return None


def resolve_selection(args):
    context = load_context(args.config, args.backend)
    if context is None:
        return None
    kernel_file = select_source(context, "kernel", args.kernel)
    host_file = select_source(context, "host", args.host)
    if kernel_file is None or host_file is None:
        return None
    context["kernel_file"] = kernel_file
    context["host_file"] = host_file
    context["selection"] = f"{Path(kernel_file).stem}__{Path(host_file).stem}"
    return context


def cmake_definitions(options):
    result = []
    for key, value in (options or {}).items():
        if isinstance(value, bool):
            value = "ON" if value else "OFF"
        result.append(f"-D{key}={value}")
    return result


def source_definitions(context, section, *, include_tests=False):
    backend = context["backend"]
    values = {}
    for role, filename in (("kernel", context["kernel_file"]), ("host", context["host_file"])):
        source_config = backend.get(role, {})
        variable = source_config.get("cmake_variable")
        directory_variable = source_config.get("dir_cmake_variable")
        if variable:
            values[variable] = filename
        if directory_variable:
            values[directory_variable] = str(resolve_path(context["project_root"], source_config["dir"]))
    if include_tests:
        tests = backend.get("tests", {})
        for key, variable in tests.get("cmake_variables", {}).items():
            if key == "memory_results":
                value = tests.get("results", {}).get("memory")
            else:
                value = tests.get(key)
            if value:
                values[variable] = str(resolve_path(context["project_root"], value))
    values.update(section.get("options", {}))
    return cmake_definitions(values)


def repository_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def default_build_dir(context, *, test=False, bench=False):
    paths = context["config"].get("paths", {})
    build_root = repository_path(paths.get("build_root", "build"))
    name = context["config"]["name"]
    backend = context["backend_name"]
    if bench:
        return build_root / name / "bench" / backend / context["selection"]
    if test:
        return build_root / name / "tests" / backend / context["selection"]
    return build_root / name / backend / context["selection"]


def build_backend(args):
    context = resolve_selection(args)
    if context is None:
        return 2
    section = context["backend"].get("build", {})
    source_dir = get_cmake_dir(context["project_root"], section.get("project", "."))
    build_dir = repository_path(args.build_dir) if args.build_dir else default_build_dir(context)
    options = dict(section.get("options", {}))
    examples_option = section.get("examples_option")
    if examples_option:
        options[examples_option] = args.examples
    command = ["cmake", "-S", str(source_dir), "-B", str(build_dir),
               *source_definitions(context, {"options": options})]
    print(f"Configuring {context['config']['name']} backend={context['backend_name']} "
          f"kernel={context['kernel_file']} host={context['host_file']} in {build_dir}")
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
    context = resolve_selection(args)
    if context is None:
        return 2
    tests = context["backend"].get("tests", {})
    if not tests.get("project"):
        print(f"ERROR: no test project configured for {context['backend_name']} in operator.yaml.", file=sys.stderr)
        return 2
    test_project = get_cmake_dir(context["project_root"], tests["project"])
    build_dir = repository_path(args.build_dir) if args.build_dir else default_build_dir(context, test=True)
    command = ["cmake", "-S", str(test_project), "-B", str(build_dir),
               *source_definitions(context, tests, include_tests=True)]
    configured = run(command, missing_tool_message=
                     "ERROR: required build tool was not found. Install CMake 3.21+ and the selected backend toolchain.")
    if configured is None:
        return 127
    if configured.returncode != 0:
        print(f"Test project configure failed for {context['config']['name']} {context['backend_name']}.",
              file=sys.stderr)
        return configured.returncode
    built = run(["cmake", "--build", str(build_dir), "--config", "Release", "--parallel"])
    if built is None:
        return 127
    if built.returncode != 0:
        print(f"Test project build failed for {context['config']['name']} {context['backend_name']}.",
              file=sys.stderr)
        return built.returncode
    tested = run(["ctest", "--test-dir", str(build_dir), "-C", "Release", "--output-on-failure"],
                 missing_tool_message="ERROR: CTest was not found; install CMake with CTest support.")
    if tested is None:
        return 127
    if tested.returncode != 0:
        print(f"Tests failed for {context['config']['name']} {context['backend_name']}.", file=sys.stderr)
        return tested.returncode
    print(f"CTest completed for {context['config']['name']} kernel={context['kernel_file']} "
          f"host={context['host_file']}; skipped suites are not passes.")
    return 0


def positive_int(value):
    try:
        number = int(value)
        if number <= 0:
            raise ValueError
        return number
    except ValueError as error:
        raise argparse.ArgumentTypeError("provide a positive integer") from error


def positive_ints(value):
    try:
        values = [int(part.strip()) for part in value.split(",")]
        if not values or any(number <= 0 for number in values):
            raise ValueError
        return values
    except ValueError as error:
        raise argparse.ArgumentTypeError("provide comma-separated positive integers") from error


def run_bench(args):
    context = resolve_selection(args)
    if context is None:
        return 2
    bench = context["backend"].get("bench")
    if not bench:
        print(f"ERROR: bench is not configured for backend {context['backend_name']}.", file=sys.stderr)
        return 2
    protocol = bench.get("protocol")
    profile = args.backend_profile
    if protocol != profile["benchmark_protocol"]:
        print(f"ERROR: {context['backend_name']} expects bench protocol "
              f"{profile['benchmark_protocol']!r}, found {protocol!r} in operator.yaml.", file=sys.stderr)
        return 2
    tests = context["backend"].get("tests", {})
    if not tests.get("project"):
        print("ERROR: bench requires a configured test project.", file=sys.stderr)
        return 2
    iterations = args.iterations if args.iterations is not None else bench.get("iterations", 100)
    sizes = args.sizes if args.sizes is not None else bench.get("sizes", [1024, 1048576])
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations <= 0:
        print("ERROR: bench.iterations in operator.yaml must be a positive integer.", file=sys.stderr)
        return 2
    if (not isinstance(sizes, list) or not sizes or
            any(not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in sizes)):
        print("ERROR: bench.sizes in operator.yaml must be a non-empty list of positive integers.",
              file=sys.stderr)
        return 2
    if any(size > bench.get("max_input", 2147483647) for size in sizes):
        print(f"ERROR: input size exceeds configured maximum {bench.get('max_input', 2147483647)}.",
              file=sys.stderr)
        return 2

    build_dir = repository_path(args.build_dir) if args.build_dir else default_build_dir(context, bench=True)
    test_project = get_cmake_dir(context["project_root"], tests["project"])
    options = dict(tests.get("options", {}))
    options.update(bench.get("options", {}))
    command = ["cmake", "-S", str(test_project), "-B", str(build_dir),
               *source_definitions(context, {"options": options}, include_tests=True)]
    configured = run(command, missing_tool_message=
                     "ERROR: CMake and the selected backend toolchain are required to build this bench.")
    if configured is None:
        return 127
    if configured.returncode != 0:
        return configured.returncode
    target = bench.get("target")
    built = run(["cmake", "--build", str(build_dir), "--config", "Release", "--target", target, "--parallel"])
    if built is None:
        return 127
    if built.returncode != 0:
        return built.returncode

    executable = args.executable
    if executable:
        executable = repository_path(executable)
    else:
        executable = build_dir / bench["executable"]
        if platform.system() == "Windows":
            executable = executable.with_suffix(".exe")
        if not executable.exists():
            executable = build_dir / "Release" / executable.name
    if not executable.is_file():
        print(f"ERROR: evaluation executable was not found: {executable}", file=sys.stderr)
        return 2

    compiler_version = "unavailable"
    compiler = shutil.which(bench.get("compiler", profile["compiler"]))
    if compiler:
        result = run([compiler, "--version"], capture=True)
        if result is not None and result.returncode == 0:
            lines = [line.strip() for line in (result.stdout or result.stderr).splitlines()
                     if "release" in line.lower()]
            if lines:
                compiler_version = lines[-1]
            else:
                compiler_version = next((line.strip() for line in
                                         (result.stdout or result.stderr).splitlines()
                                         if line.strip()), "unavailable")

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for size in sizes:
        measured = run([str(executable), str(size), str(iterations)], capture=True)
        if measured is None:
            return 127
        fields = measured.stdout.strip().split("\t")
        expected_tag = profile["result_tag"]
        if len(fields) != 16 or fields[0] != expected_tag:
            print(measured.stderr or f"ERROR: unexpected evaluation output: {measured.stdout!r}", file=sys.stderr)
            return measured.returncode or 2
        try:
            measured_kernel, measured_host, device, runtime_version, driver_version = fields[1:6]
            measured_size, working_set = int(fields[6]), int(fields[7])
            memory_free_before, memory_total = int(fields[8]), int(fields[9])
            memory_free_after_alloc, memory_free_after_release = int(fields[10]), int(fields[11])
            allocation_status = fields[12]
            average_latency, bandwidth = float(fields[13]), float(fields[14])
            validation = fields[15]
        except ValueError:
            print(f"ERROR: malformed evaluation output: {measured.stdout!r}", file=sys.stderr)
            return 2
        if (measured_kernel != context["kernel_file"] or measured_host != context["host_file"] or
                measured_size != size):
            print("ERROR: evaluation executable reported a different source selection or input size.",
                  file=sys.stderr)
            return 2
        allocation_ok = allocation_status == "allocated"
        if allocation_ok and (measured.returncode != 0 or average_latency <= 0 or bandwidth <= 0 or validation != "passed"):
            print(f"ERROR: evaluation failed for N={size}: {validation}", file=sys.stderr)
            return measured.returncode or 1
        rows.append({
            "timestamp_utc": timestamp, "operator": context["config"]["name"],
            "backend": context["backend_name"], "kernel_file": context["kernel_file"],
            "host_file": context["host_file"], "device": device, "input_size": measured_size,
            "dtype": bench.get("dtype", "float32"),
            "compiler": bench.get("compiler", profile["compiler"]),
            "compiler_version": compiler_version,
            "runtime": profile["runtime_format"].format(
                runtime_version=runtime_version, driver_version=driver_version),
            "iterations": iterations, "working_set_bytes": working_set,
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
            print(measured.stderr or f"Evaluation exited {measured.returncode}", file=sys.stderr)
            return measured.returncode

    result_dir = resolve_path(context["project_root"], tests["results"]["bench"])
    output = args.output or result_dir / f"{context['selection']}.csv"
    output = repository_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output.exists() or output.stat().st_size == 0
    with output.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"Recorded {len(rows)} real evaluation row(s) in {output}")
    return 0


def package_backend(args):
    context = resolve_selection(args)
    if context is None:
        return 2
    package_config = context["backend"].get("package", {})
    if not package_config:
        print(f"ERROR: packaging is not configured for {context['backend_name']}.", file=sys.stderr)
        return 2
    files = list(package_config.get("include", []))
    for role, filename in (("kernel", context["kernel_file"]), ("host", context["host_file"])):
        directory = context["backend"][role]["dir"]
        files.append((Path(directory) / filename).as_posix())
    if package_config.get("include_all_sources"):
        for role in ("kernel", "host"):
            directory = context["backend"][role]["dir"]
            files.extend((Path(directory) / source.name).as_posix()
                         for source in configured_sources(context, role))
    bench_results = context["backend"].get("tests", {}).get("results", {}).get("bench")
    if bench_results:
        result_file = f"{bench_results}/{context['selection']}.csv"
        if resolve_path(context["project_root"], result_file).is_file():
            files.append(result_file)
    files = list(dict.fromkeys(files))
    for relative in files:
        if not resolve_path(context["project_root"], relative).is_file():
            print(f"ERROR: required package file is missing: {relative}", file=sys.stderr)
            return 2

    config = context["config"]
    dist_root = repository_path(config.get("paths", {}).get("dist_root", "dist"))
    selection = context["selection"]
    destination = args.output or dist_root / f"{config['name']}-{config.get('version', '0.0.0')}-{context['backend_name']}-{selection}"
    destination = repository_path(destination)
    if destination.exists():
        if not args.force:
            print(f"ERROR: {destination} already exists; pass --force to replace a recognized generated package.",
                  file=sys.stderr)
            return 2
        old_manifest = destination / "package.json"
        try:
            previous = json.loads(old_manifest.read_text(encoding="utf-8"))
            if previous.get("operator") != config["name"]:
                raise ValueError("package belongs to a different operator")
            old_files = previous["files"]
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"ERROR: refusing to replace unrecognized directory {destination}: {error}", file=sys.stderr)
            return 2
        for relative in old_files + ["package.json"]:
            target = (destination / relative).resolve()
            if not target.is_relative_to(destination.resolve()):
                print(f"ERROR: unsafe path in existing package manifest: {relative}", file=sys.stderr)
                return 2
            if target.is_file() or target.is_symlink():
                target.unlink()

    for relative in files:
        source = resolve_path(context["project_root"], relative)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    manifest = {
        "operator": config["name"], "version": config.get("version", "0.0.0"),
        "backend": context["backend_name"], "kernel_file": context["kernel_file"],
        "host_file": context["host_file"], "files": files,
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "package.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                               encoding="utf-8")
    print(f"Packaged {config['name']} into {destination}")
    return 0


def main(backend_name, backend_profile):
    args = make_parser(backend_name, backend_profile).parse_args()
    environment_status = check_backend_environment(backend_profile)
    if environment_status != 0:
        return environment_status
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit("Run tools/operator_cuda.py or tools/operator_cann.py instead.")
