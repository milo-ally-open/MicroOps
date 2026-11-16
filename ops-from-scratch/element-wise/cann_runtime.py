"""Small PyACL + ctypes runtime for temporary CANN kernel demos."""
from __future__ import annotations
import ctypes
import os
import shutil
import subprocess
from pathlib import Path
import numpy as np


def cann_root() -> Path:
    for key in ("ASCEND_HOME_PATH", "ASCEND_HOME_DIR", "ASCEND_CANN_PACKAGE_PATH"):
        value = os.environ.get(key)
        if value:
            root = Path(value).resolve()
            if (root / "aarch64-linux/tikcpp/tikcfw/kernel_operator.h").is_file():
                return root
    root = Path("/usr/local/Ascend/ascend-toolkit/latest").resolve()
    if (root / "aarch64-linux/tikcpp/tikcfw/kernel_operator.h").is_file():
        return root
    raise RuntimeError("CANN headers not found. Source the board's CANN set_env.sh first.")


def build_kernel(source: Path, output_dir: Path) -> ctypes.CDLL:
    root = cann_root()
    compiler = shutil.which("ccec")
    if not compiler:
        raise RuntimeError("ccec was not found in PATH. Source CANN set_env.sh first.")
    fw = root / "aarch64-linux/tikcpp/tikcfw"
    output = output_dir / f"lib{source.stem}.so"
    rpaths = [root / "aarch64-linux/lib64", root / "runtime/lib64",
              root / "lib64", Path("/var/davinci/driver/lib64")]
    rpath = ":".join(str(p) for p in rpaths if p.is_dir())
    command = [
        compiler, "-x", "cce", str(source), "--cce-aicore-arch=dav-c220",
        "-std=c++17", "-O2", "-fPIC", "-shared",
        f"-I{fw}", f"-I{fw / 'interface'}", f"-I{fw / 'impl'}",
        f"-I{fw / 'impl/dav_c220'}", f"-I{root / 'aarch64-linux/include'}",
        f"-L{root / 'aarch64-linux/lib64'}", f"-L{root / 'runtime/lib64'}",
        "-lruntime", "-lascendcl", f"-Wl,-rpath,{rpath}", "-o", str(output),
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError("CANN kernel compilation failed:\n" + result.stdout + result.stderr)
    return ctypes.CDLL(str(output))


class PyACLSession:
    """Own one ACL device/context/stream and its temporary device allocations."""
    def __init__(self, device_id: int = 0):
        try:
            import acl
        except ImportError as error:
            raise RuntimeError("PyACL is unavailable; use the Python environment shipped with CANN.") from error
        self.acl, self.device_id = acl, device_id
        self.allocations = []
        self.context = self.stream = None
        self.initialized = False

    @staticmethod
    def check(status, operation):
        if status != 0:
            raise RuntimeError(f"{operation} failed with ACL status {status}")

    def __enter__(self):
        self.check(self.acl.init(), "acl.init")
        self.initialized = True
        self.check(self.acl.rt.set_device(self.device_id), "acl.rt.set_device")
        self.context, status = self.acl.rt.create_context(self.device_id)
        self.check(status, "acl.rt.create_context")
        self.stream, status = self.acl.rt.create_stream()
        self.check(status, "acl.rt.create_stream")
        return self

    def to_device(self, array: np.ndarray) -> int:
        array = np.ascontiguousarray(array)
        pointer, status = self.acl.rt.malloc(array.nbytes, 0)
        self.check(status, "acl.rt.malloc")
        self.allocations.append(pointer)
        if array.nbytes:
            self.check(self.acl.rt.memcpy(pointer, array.nbytes, array.ctypes.data, array.nbytes, 1),
                       "host-to-device copy")
        return pointer

    def from_device(self, pointer: int, result: np.ndarray) -> np.ndarray:
        result = np.ascontiguousarray(result)
        if result.nbytes:
            self.check(self.acl.rt.memcpy(result.ctypes.data, result.nbytes, pointer, result.nbytes, 2),
                       "device-to-host copy")
        return result

    def synchronize(self):
        self.check(self.acl.rt.synchronize_stream(self.stream), "stream synchronize")

    def __exit__(self, exc_type, exc_value, traceback):
        for pointer in reversed(self.allocations):
            self.acl.rt.free(pointer)
        if self.stream is not None:
            self.acl.rt.destroy_stream(self.stream)
        if self.context is not None:
            self.acl.rt.destroy_context(self.context)
        if self.initialized:
            self.acl.rt.reset_device(self.device_id)
            self.acl.finalize()
