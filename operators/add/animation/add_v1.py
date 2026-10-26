"""CUDA v1 walkthrough for elementwise float32 addition."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "skills/ops-manim/scripts"))
from elementwise_scene import ElementwiseV1Scene  # noqa: E402


class AddV1HardwareWalkthrough(ElementwiseV1Scene):
    OP_KEY = "add"
    OPERATOR = "Add · C[i] = A[i] + B[i]"
    FORMULA = "C[i] = A[i] + B[i]"
    INPUT_LABELS = ("A[0]=1, B[0]=8", "A[1]=4, B[1]=2", "A[2]=2, B[2]=9", "A[3]=10, B[3]=20")
    OUTPUTS = ("C[0]=9", "C[1]=6", "C[2]=11", "C[3]=30")
    CODE = "C[idx] = A[idx] + B[idx];"
    BINARY = True
    CUDA_DISPATCH_THRESHOLD = 262144
    TRITON_SMALL_BLOCK = 512
    TRITON_LARGE_BLOCK = 1024
    TRITON_THRESHOLD = 8388608
