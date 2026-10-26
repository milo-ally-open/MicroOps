"""CUDA v1 walkthrough for the tanh activation function."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "skills/ops-manim/scripts"))
from elementwise_scene import ElementwiseV1Scene  # noqa: E402


class TanhV1Walkthrough(ElementwiseV1Scene):
    OPERATOR = "tanh"
    FORMULA = "Y[i] = tanh(X[i])"
    INPUT_LABELS = ("X[0]=-2", "X[1]=-0.5", "X[2]=0.5", "X[3]=2")
    OUTPUTS = ("Y[0]≈-0.9640", "Y[1]≈-0.4621", "Y[2]≈0.4621", "Y[3]≈0.9640")
    CODE = "Y[idx] = tanhf(x);"
    GRAPH_KEY = "tanh"
    TRITON_SMALL_BLOCK = 512
    TRITON_LARGE_BLOCK = 512
    TRITON_THRESHOLD = 8388608
    GRAPH_Y_RANGE = (-1.2, 1.2)
    GRAPH_NOTE = "输出被压缩到 (-1, 1)；输入绝对值增大时曲线逐渐饱和。"
