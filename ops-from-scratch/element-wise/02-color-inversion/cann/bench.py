"""Temporary CANN build, correctness test, and benchmark for RGBA inversion."""
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
    pointer, status = acl.rt.malloc(array.nbytes, 0)
    check(status, "acl.rt.malloc")
    allocations.append(pointer)
    check(acl.rt.memcpy(pointer, array.nbytes, array.ctypes.data, array.nbytes, 1),
          "host-to-device copy")
    return pointer


def copy_to_host(pointer, result):
    check(acl.rt.memcpy(result.ctypes.data, result.nbytes, pointer, result.nbytes, 2),
          "device-to-host copy")
    return result


def print_environment(height, width, size_bytes):
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
    print(f"input: shape=({height},{width},4); dtype=uint8; device buffer bytes={size_bytes}")


def expected(image):
    result = image.copy()
    result[..., :3] = 255 - result[..., :3]
    return result


def main():
    rng = np.random.default_rng(11)
    allocations = []
    initialized = False
    context = stream = None
    with tempfile.TemporaryDirectory(prefix="microops-color-inversion-") as temp:
        library_path = Path(temp) / "libcolor_inversion.so"
        compile_kernel(library_path)
        library = ctypes.CDLL(str(library_path))
        launch = library.color_inversion_launch
        launch.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
                           ctypes.c_void_p]
        launch.restype = None
        check(acl.init(), "acl.init")
        initialized = True
        try:
            check(acl.rt.set_device(DEVICE_ID), "acl.rt.set_device")
            context, status = acl.rt.create_context(DEVICE_ID)
            check(status, "acl.rt.create_context")
            stream, status = acl.rt.create_stream()
            check(status, "acl.rt.create_stream")

            for height, width in ((1, 1), (2, 3), (33, 65)):
                image = rng.integers(0, 256, (height, width, 4), dtype=np.uint8)
                dimage = device_buffer(image, allocations)
                launch(dimage, width, height, stream)
                check(acl.rt.synchronize_stream(stream), "stream synchronize")
                np.testing.assert_array_equal(
                    copy_to_host(dimage, np.empty_like(image)), expected(image)
                )
            print("test successfully! shapes: (1,1,4), (2,3,4), (33,65,4); alpha preserved")

            height, width, iterations = 512, 512, 20
            image = rng.integers(0, 256, (height, width, 4), dtype=np.uint8)
            dimage = device_buffer(image, allocations)
            launch(dimage, width, height, stream)
            check(acl.rt.synchronize_stream(stream), "warmup synchronize")
            start = time.perf_counter()
            for _ in range(iterations):
                launch(dimage, width, height, stream)
            check(acl.rt.synchronize_stream(stream), "benchmark synchronize")
            elapsed = time.perf_counter() - start
            print_environment(height, width, image.nbytes)
            print(f"host launch+completion: {elapsed * 1e6 / iterations:.3f} us/op")
            print(f"estimated read+write bandwidth: "
                  f"{8 * width * height * iterations / elapsed / 1e9:.3f} GB/s")
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
