"""Temporary CANN build, correctness test, and benchmark for ReLU."""
from __future__ import annotations

import ctypes
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

try:
    import acl
except ImportError as error:
    raise SystemExit("PyACL is unavailable. Source the board's CANN set_env.sh first.") from error

SOURCE = Path(__file__).with_name("solve.cpp")
CANN_ROOT = Path(os.environ.get(
    "ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"
)).resolve()
DEVICE_ID = 0


def check(status, operation):
    if status != 0:
        raise RuntimeError(f"{operation} failed with ACL status {status}")


def compile_kernel(output):
    compiler = shutil.which("ccec")
    framework = CANN_ROOT / "aarch64-linux/tikcpp/tikcfw"
    if not compiler or not (framework / "kernel_operator.h").is_file():
        raise SystemExit("CANN compiler or Ascend C headers are missing; source set_env.sh.")
    libraries = ":".join(str(path) for path in (
        CANN_ROOT / "aarch64-linux/lib64", CANN_ROOT / "runtime/lib64",
        CANN_ROOT / "lib64", Path("/var/davinci/driver/lib64")
    ) if path.is_dir())
    command = [
        compiler, "-x", "cce", str(SOURCE), "--cce-aicore-arch=dav-c220",
        "-std=c++17", "-O2", "-fPIC", "-shared",
        f"-I{framework}", f"-I{framework / 'interface'}",
        f"-I{framework / 'impl'}", f"-I{framework / 'impl/dav_c220'}",
        f"-I{CANN_ROOT / 'aarch64-linux/include'}",
        f"-L{CANN_ROOT / 'aarch64-linux/lib64'}",
        f"-L{CANN_ROOT / 'runtime/lib64'}",
        "-lruntime", "-lascendcl", f"-Wl,-rpath,{libraries}", "-o", str(output),
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError("CANN compilation failed:\n" + result.stdout + result.stderr)


def device_buffer(array, allocations):
    array = np.ascontiguousarray(array)
    if not array.nbytes:
        return 0
    pointer, status = acl.rt.malloc(array.nbytes, 0)
    check(status, "acl.rt.malloc")
    allocations.append(pointer)
    check(acl.rt.memcpy(pointer, array.nbytes, array.ctypes.data, array.nbytes, 1),
          "host-to-device copy")
    return pointer


def copy_to_host(pointer, result):
    if result.nbytes:
        check(acl.rt.memcpy(result.ctypes.data, result.nbytes, pointer, result.nbytes, 2),
              "device-to-host copy")
    return result


def print_environment(count):
    compiler = shutil.which("ccec") or "unavailable"
    version = subprocess.run([compiler, "--version"], text=True, capture_output=True).stdout
    smi = subprocess.run(["npu-smi", "info"], text=True, capture_output=True) if shutil.which("npu-smi") else None
    info = smi.stdout if smi else ""
    match = re.search(r"^\|\s*0\s+(\S+)\s+\|", info, re.MULTILINE)
    driver = next((line.strip().strip("|").strip() for line in info.splitlines()
                   if "npu-smi " in line), "unavailable")
    print(f"NPU={match.group(1) if match else 'NPU'} (id={DEVICE_ID}, {driver}); "
          f"CANN={CANN_ROOT.name}; ccec={version.splitlines()[0] if version else 'unknown'}")
    print(f"Input: N={count}, float32, {count * 2 * 4} device bytes")


def main():
    allocations = []
    initialized = False
    context = stream = None
    with tempfile.TemporaryDirectory(prefix="microops-relu-") as temp:
        library_path = Path(temp) / "librelu.so"
        print("Compiling CANN kernel...", flush=True)
        compile_kernel(library_path)
        print("CANN kernel ready.", flush=True)
        library = ctypes.CDLL(str(library_path))
        launch = library.relu_launch
        launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
                           ctypes.c_void_p]
        launch.restype = None

        print("Initializing ACL...", flush=True)
        check(acl.init(), "acl.init")
        initialized = True
        try:
            check(acl.rt.set_device(DEVICE_ID), "acl.rt.set_device")
            context, status = acl.rt.create_context(DEVICE_ID)
            check(status, "acl.rt.create_context")
            stream, status = acl.rt.create_stream()
            check(status, "acl.rt.create_stream")
            print("Testing correctness...", flush=True)
            for count in (0, 1, 3, 4, 5, 257, 4099):
                input = ((np.arange(count, dtype=np.int32) % 11) - 5).astype(np.float32)
                output = np.empty_like(input)
                dinput = device_buffer(input, allocations)
                doutput = device_buffer(output, allocations)
                launch(dinput, doutput, count, stream)
                if count == 0:
                    continue
                check(acl.rt.synchronize_stream(stream), "stream synchronize")
                expected = np.maximum(input, np.float32(0.0))
                np.testing.assert_array_equal(copy_to_host(doutput, output), expected)
            print("Tests passed (N=0, 1, 3, 4, 5, 257, 4099).")

            count, iterations = 65536, 100
            input = ((np.arange(count, dtype=np.int32) % 257) - 128).astype(np.float32)
            output = np.empty_like(input)
            dinput = device_buffer(input, allocations)
            doutput = device_buffer(output, allocations)
            launch(dinput, doutput, count, stream)
            print_environment(count)
            check(acl.rt.synchronize_stream(stream), "warmup synchronize")
            print(f"Benchmark (N={count}, float32, {iterations} iterations):", flush=True)
            start = time.perf_counter()
            for _ in range(iterations):
                launch(dinput, doutput, count, stream)
            check(acl.rt.synchronize_stream(stream), "benchmark synchronize")
            elapsed = time.perf_counter() - start
            print(f"  {elapsed * 1e6 / iterations:.3f} us/op; "
                  f"estimated bandwidth {2 * count * 4 * iterations / elapsed / 1e9:.3f} GB/s")
        finally:
            for pointer in reversed(allocations):
                acl.rt.free(pointer)
            if stream is not None:
                acl.rt.destroy_stream(stream)
            if context is not None:
                acl.rt.destroy_context(context)
            acl.rt.reset_device(DEVICE_ID)
            if initialized:
                acl.finalize()


if __name__ == "__main__":
    main()
