"""CUDA v1 walkthrough for ReLU, including its piecewise graph."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "skills/ops-manim/scripts"))
from elementwise_scene import ElementwiseV1Scene  # noqa: E402


class ReluV1Walkthrough(ElementwiseV1Scene):
    OPERATOR = "ReLU"
    FORMULA = "Y[i] = max(X[i], 0)   (NaN is preserved)"
    INPUT_LABELS = ("X[0]=-2", "X[1]=-0.5", "X[2]=0.5", "X[3]=2")
    OUTPUTS = ("Y[0]=0", "Y[1]=0", "Y[2]=0.5", "Y[3]=2")
    CODE = "Y[idx] = (x > 0.0f ? x : (x != x ? x : 0.0f));"
    GRAPH_KEY = "relu"
    GRAPH_Y_RANGE = (-0.5, 4.5)
    GRAPH_NOTE = "x < 0 映射到 0；x > 0 保持原值。当前实现显式保留 NaN。"
