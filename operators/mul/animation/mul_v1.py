"""CUDA v1 walkthrough for elementwise float32 multiplication."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "skills/ops-manim/scripts"))
from elementwise_scene import ElementwiseV1Scene  # noqa: E402


class MulV1Walkthrough(ElementwiseV1Scene):
    OP_KEY = "mul"
    OPERATOR = "Multiplication · C[i] = A[i] × B[i]"
    FORMULA = "C[i] = A[i] × B[i]"
    INPUT_LABELS = ("A[0]=2, B[0]=3", "A[1]=-3, B[1]=2", "A[2]=0.5, B[2]=-3", "A[3]=4, B[3]=0.25")
    OUTPUTS = ("C[0]=6", "C[1]=-6", "C[2]=-1.5", "C[3]=1")
    CODE = "C[idx] = A[idx] * B[idx];"
    BINARY = True
    CUDA_DISPATCH_THRESHOLD = 262144
    TRITON_SMALL_BLOCK = 512
    TRITON_LARGE_BLOCK = 2048
    TRITON_THRESHOLD = 8388608
