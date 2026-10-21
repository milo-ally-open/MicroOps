"""CUDA v1 walkthrough for numerically stable sigmoid evaluation."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "skills/ops-manim/scripts"))
from elementwise_scene import ElementwiseV1Scene  # noqa: E402


class SigmoidV1Walkthrough(ElementwiseV1Scene):
    OPERATOR = "Sigmoid"
    FORMULA = "y = 1 / (1 + exp(-x))"
    INPUT_LABELS = ("X[0]=-2", "X[1]=-0.5", "X[2]=0.5", "X[3]=2")
    OUTPUTS = ("Y[0]≈0.1192", "Y[1]≈0.3775", "Y[2]≈0.6225", "Y[3]≈0.8808")
    CODE = "x>=0 ? 1/(1+expf(-x)) : expf(x)/(1+expf(x))"
    GRAPH_KEY = "sigmoid"
    GRAPH_Y_RANGE = (-0.2, 1.2)
    GRAPH_NOTE = "曲线是 sigmoid；CUDA kernel 按 x 的符号选择等价形式，避免计算大幅正指数。"
