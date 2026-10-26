"""CUDA v1 walkthrough for the SiLU activation function."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "skills/ops-manim/scripts"))
from elementwise_scene import ElementwiseV1Scene  # noqa: E402


class SiluV1Walkthrough(ElementwiseV1Scene):
    OPERATOR = "SiLU · x × sigmoid(x)"
    FORMULA = "Y[i] = X[i] / (1 + exp(-X[i]))"
    INPUT_LABELS = ("X[0]=-2", "X[1]=-0.5", "X[2]=0.5", "X[3]=2")
    OUTPUTS = ("Y[0]≈-0.2384", "Y[1]≈-0.1888", "Y[2]≈0.3112", "Y[3]≈1.7616")
    CODE = "Y[idx] = x / (1.0f + expf(-x));"
    GRAPH_KEY = "silu"
    TRITON_SMALL_BLOCK = 512
    TRITON_LARGE_BLOCK = 512
    TRITON_THRESHOLD = 8388608
    GRAPH_Y_RANGE = (-1.0, 4.5)
    GRAPH_NOTE = "SiLU = x·sigmoid(x)：负区间保留小幅负值，正区间逐渐接近恒等映射。"
