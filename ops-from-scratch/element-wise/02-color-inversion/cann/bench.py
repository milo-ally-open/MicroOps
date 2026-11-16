"""Compile, test, and benchmark color_inversion using PyACL and a temporary library."""
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


def expected(image):
    result = image.copy()
    result[..., :3] = 255 - result[..., :3]
    return result


def main():
    rng = np.random.default_rng(11)
    with tempfile.TemporaryDirectory(prefix="microops-color-inversion-") as temp:
        library = build_kernel(SOURCE, Path(temp))
        launch = library.color_inversion_launch
        launch.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
        with PyACLSession(0) as device:
            for height, width in ((1, 1), (2, 3), (33, 65)):
                image = rng.integers(0, 256, (height, width, 4), dtype=np.uint8)
                dimage = device.to_device(image)
                launch(dimage, width, height, device.stream)
                device.synchronize()
                np.testing.assert_array_equal(device.from_device(dimage, np.empty_like(image)),
                                              expected(image))
            print("test successfully! shapes: (1,1,4), (2,3,4), (33,65,4); alpha preserved")

            height, width, iterations = 512, 512, 20
            image = rng.integers(0, 256, (height, width, 4), dtype=np.uint8)
            dimage = device.to_device(image)
            launch(dimage, width, height, device.stream)
            device.synchronize()
            start = time.perf_counter()
            for _ in range(iterations):
                launch(dimage, width, height, device.stream)
            device.synchronize()
            elapsed = time.perf_counter() - start
            print(f"device: Ascend NPU 0; CANN: {os.environ.get('ASCEND_HOME_PATH', 'unknown')}")
            print(f"compiler: {shutil.which('ccec')}; shape=({height},{width},4); uint8; device bytes={image.nbytes}")
            print(f"wall launch+completion: {elapsed*1e6/iterations:.3f} us/op")
            print(f"estimated read+write bandwidth: {8*width*height*iterations/elapsed/1e9:.3f} GB/s")


if __name__ == "__main__":
    main()
