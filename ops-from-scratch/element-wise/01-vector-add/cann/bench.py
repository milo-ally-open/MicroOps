"""Temporary CANN build, correctness test, and benchmark for vector addition."""
from __future__ import annotations

import ctypes
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

_script_start = time.perf_counter()
print("[start] vector add CANN benchmark; loading NumPy and PyACL", flush=True)

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
    result = subprocess.run(command, text=True)
    if result.returncode:
        raise RuntimeError(f"CANN compilation failed with exit code {result.returncode}")


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
    print(f"device: {match.group(1) if match else 'NPU'}; id={DEVICE_ID}; {driver}")
    print(f"CANN: {CANN_ROOT.name}; compiler: {compiler}; "
          f"{version.splitlines()[0] if version else 'version unavailable'}")
    print(f"input: N={count}; dtype=float32; device buffer bytes={count * 3 * 4}")


def main():
    rng = np.random.default_rng(7)
    allocations = []
    initialized = False
    context = stream = None
    with tempfile.TemporaryDirectory(prefix="microops-vector-add-") as temp:
        library_path = Path(temp) / "libvector_add.so"
        print("[compile] compiling Ascend C kernel (compiler output follows)", flush=True)
        compile_start = time.perf_counter()
        compile_kernel(library_path)
        print(f"[compile] ready in {time.perf_counter() - compile_start:.2f}s", flush=True)
        library = ctypes.CDLL(str(library_path))
        launch = library.vector_add_launch
        launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                           ctypes.c_uint32, ctypes.c_void_p]
        launch.restype = None
        print("[runtime] initializing ACL and creating device context/stream", flush=True)
        runtime_start = time.perf_counter()
        check(acl.init(), "acl.init")
        initialized = True
        try:
            check(acl.rt.set_device(DEVICE_ID), "acl.rt.set_device")
            context, status = acl.rt.create_context(DEVICE_ID)
            check(status, "acl.rt.create_context")
            stream, status = acl.rt.create_stream()
            check(status, "acl.rt.create_stream")
            print(f"[runtime] ready in {time.perf_counter() - runtime_start:.2f}s", flush=True)

            for count in (0, 1, 3, 4, 257, 4099):
                print(f"[test] checking N={count}", flush=True)
                a = rng.standard_normal(count).astype(np.float32)
                b = rng.standard_normal(count).astype(np.float32)
                if count == 0:
                    launch(0, 0, 0, 0, stream)
                    continue
                da, db, dc = (device_buffer(x, allocations)
                              for x in (a, b, np.empty_like(a)))
                launch(da, db, dc, count, stream)
                check(acl.rt.synchronize_stream(stream), "stream synchronize")
                np.testing.assert_allclose(copy_to_host(dc, np.empty_like(a)), a + b,
                                           rtol=1e-6, atol=1e-6)
            print("[test] passed sizes: 0, 1, 3, 4, 257, 4099", flush=True)

            count, iterations = 65536, 100
            a = rng.standard_normal(count).astype(np.float32)
            b = rng.standard_normal(count).astype(np.float32)
            da, db, dc = (device_buffer(x, allocations)
                          for x in (a, b, np.empty_like(a)))
            launch(da, db, dc, count, stream)
            print("[environment] collecting device, compiler, and input details", flush=True)
            print_environment(count)
            print(f"[bench] warmup N={count}; iterations={iterations}", flush=True)
            check(acl.rt.synchronize_stream(stream), "warmup synchronize")
            print("[bench] warmup complete; timing launches", flush=True)
            start = time.perf_counter()
            for _ in range(iterations):
                launch(da, db, dc, count, stream)
            check(acl.rt.synchronize_stream(stream), "benchmark synchronize")
            elapsed = time.perf_counter() - start
            print(f"[bench] complete in {elapsed:.3f}s; host launch+completion: {elapsed * 1e6 / iterations:.3f} us/op", flush=True)
            print(f"[bench] estimated read+write bandwidth: {3 * count * 4 * iterations / elapsed / 1e9:.3f} GB/s", flush=True)
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
            print(f"[done] total elapsed {time.perf_counter() - _script_start:.2f}s", flush=True)


if __name__ == "__main__":
    main()
