"""Teaching scene based on src/cuda/kernel/relu_kernel_v1.cu."""
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
    TEACHING_NOTE = "负数映射到 0；当前实现保留 NaN。"
