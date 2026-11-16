"""Compile, test, and benchmark vector_add using PyACL and a temporary shared library."""
from __future__ import annotations
import ctypes
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cann_runtime import PyACLSession, build_kernel

SOURCE = Path(__file__).with_name("solve.cpp")


def main():
    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory(prefix="microops-vector-add-") as temp:
        library = build_kernel(SOURCE, Path(temp))
        launch = library.vector_add_launch
        launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                           ctypes.c_uint32, ctypes.c_void_p]
        with PyACLSession(0) as device:
            for n in (0, 1, 3, 4, 257, 4099):
                if n == 0:
                    launch(0, 0, 0, 0, device.stream)
                    continue
                a = rng.standard_normal(n).astype(np.float32)
                b = rng.standard_normal(n).astype(np.float32)
                da, db, dc = (device.to_device(x) for x in (a, b, np.empty_like(a)))
                launch(da, db, dc, n, device.stream)
                device.synchronize()
                np.testing.assert_allclose(device.from_device(dc, np.empty_like(a)), a + b,
                                           rtol=1e-6, atol=1e-6)
            print("test successfully! sizes: 0, 1, 3, 4, 257, 4099")

            n, iterations = 65536, 100
            a = rng.standard_normal(n).astype(np.float32)
            b = rng.standard_normal(n).astype(np.float32)
            da, db, dc = (device.to_device(x) for x in (a, b, np.empty_like(a)))
            launch(da, db, dc, n, device.stream)
            device.synchronize()
            start = time.perf_counter()
            for _ in range(iterations):
                launch(da, db, dc, n, device.stream)
            device.synchronize()
            elapsed = time.perf_counter() - start
            print(f"device: Ascend NPU 0; CANN: {os.environ.get('ASCEND_HOME_PATH', 'unknown')}")
            print(f"compiler: {shutil.which('ccec')}; N={n}; float32; device bytes={3*n*4}")
            print(f"wall launch+completion: {elapsed*1e6/iterations:.3f} us/op")
            print(f"estimated read+write bandwidth: {3*n*4*iterations/elapsed/1e9:.3f} GB/s")


if __name__ == "__main__":
    main()
