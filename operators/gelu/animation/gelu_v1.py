"""CUDA v1 walkthrough for the tanh-approximate GELU function."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "skills/ops-manim/scripts"))
from elementwise_scene import ElementwiseV1Scene  # noqa: E402


class GeluV1Walkthrough(ElementwiseV1Scene):
    OPERATOR = "GELU · tanh approximation"
    FORMULA = "0.5x(1+tanh(√(2/π)(x+0.044715x³)))"
    INPUT_LABELS = ("X[0]=-2", "X[1]=-0.5", "X[2]=0.5", "X[3]=2")
    OUTPUTS = ("Y[0]≈-0.0454", "Y[1]≈-0.1543", "Y[2]≈0.3457", "Y[3]≈1.9546")
    CODE = "0.5f*x*(1.0f+tanhf(0.79788456f*(x+0.044715f*x*x*x)))"
    GRAPH_KEY = "gelu"
    GRAPH_Y_RANGE = (-0.5, 4.5)
    GRAPH_NOTE = "曲线按 CUDA v1 中的 tanh 近似公式计算，并非精确 erf 形式。"
