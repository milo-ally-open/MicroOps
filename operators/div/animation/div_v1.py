"""CUDA v1 walkthrough for elementwise float32 division."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "skills/ops-manim/scripts"))
from elementwise_scene import ElementwiseV1Scene  # noqa: E402


class DivV1Walkthrough(ElementwiseV1Scene):
    OPERATOR = "Division · C[i] = A[i] / B[i]"
    FORMULA = "C[i] = A[i] / B[i]"
    INPUT_LABELS = ("A[0]=8, B[0]=-2", "A[1]=3, B[1]=2", "A[2]=-9, B[2]=-3", "A[3]=1, B[3]=4")
    OUTPUTS = ("C[0]=-4", "C[1]=1.5", "C[2]=3", "C[3]=0.25")
    CODE = "C[idx] = A[idx] / B[idx];"
    BINARY = True
