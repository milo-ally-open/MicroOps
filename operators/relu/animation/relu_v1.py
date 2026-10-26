"""CUDA v1 walkthrough for ReLU, including its piecewise graph."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "skills/ops-manim/scripts"))
from elementwise_scene import ElementwiseV1Scene  # noqa: E402


class ReluV1Walkthrough(ElementwiseV1Scene):
    OPERATOR = "ReLU"
    FRAMEWORK_OPTIMIZATION = "relu_fusion"
    FORMULA = "Y[i] = max(X[i], 0)   (NaN is preserved)"
    INPUT_LABELS = ("X[0]=-2", "X[1]=-0.5", "X[2]=0.5", "X[3]=2")
    OUTPUTS = ("Y[0]=0", "Y[1]=0", "Y[2]=0.5", "Y[3]=2")
    CODE = "Y[idx] = (x < 0.0f ? 0.0f : x);"
    GRAPH_KEY = "relu"
    TRITON_SMALL_BLOCK = 512
    TRITON_LARGE_BLOCK = 2048
    TRITON_THRESHOLD = 8388608
    GRAPH_Y_RANGE = (-0.5, 4.5)
    GRAPH_NOTE = "x < 0 映射到 0；其他值原样返回。有序比较对 NaN 为假，因此 NaN 和 -0 都保留。"
