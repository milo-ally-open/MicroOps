"""CUDA v1 walkthrough for elementwise float32 multiplication."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "skills/ops-manim/scripts"))
from elementwise_scene import ElementwiseV1Scene  # noqa: E402


class MulV1Walkthrough(ElementwiseV1Scene):
    OPERATOR = "Multiplication · C[i] = A[i] × B[i]"
    FORMULA = "C[i] = A[i] × B[i]"
    INPUT_LABELS = ("A[0]=2, B[0]=3", "A[1]=-3, B[1]=2", "A[2]=0.5, B[2]=-8", "A[3]=4, B[3]=0.25")
    OUTPUTS = ("C[0]=6", "C[1]=-6", "C[2]=-4", "C[3]=1")
    CODE = "C[idx] = A[idx] * B[idx];"
    BINARY = True
