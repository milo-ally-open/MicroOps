"""CUDA v1 walkthrough for elementwise float32 subtraction."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "skills/ops-manim/scripts"))
from elementwise_scene import ElementwiseV1Scene  # noqa: E402


class SubV1Walkthrough(ElementwiseV1Scene):
    OP_KEY = "sub"
    OPERATOR = "Subtraction · C[i] = A[i] - B[i]"
    FORMULA = "C[i] = A[i] - B[i]"
    INPUT_LABELS = ("A[0]=5, B[0]=1", "A[1]=-2, B[1]=4", "A[2]=3.5, B[2]=-1.5", "A[3]=8, B[3]=2")
    OUTPUTS = ("C[0]=4", "C[1]=-6", "C[2]=5", "C[3]=6")
    CODE = "C[idx] = A[idx] - B[idx];"
    BINARY = True
    CUDA_DISPATCH_THRESHOLD = 262144
    TRITON_SMALL_BLOCK = 512
    TRITON_LARGE_BLOCK = 2048
    TRITON_THRESHOLD = 8388608
