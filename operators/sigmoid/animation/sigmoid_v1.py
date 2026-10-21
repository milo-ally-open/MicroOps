"""Teaching scene based on src/cuda/kernel/sigmoid_kernel_v1.cu."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "skills/ops-manim/scripts"))
from elementwise_scene import ElementwiseV1Scene  # noqa: E402


class SigmoidV1Walkthrough(ElementwiseV1Scene):
    OPERATOR = "Sigmoid"
    FORMULA = "x≥0: 1/(1+e⁻ˣ)    x<0: eˣ/(1+eˣ)"
    INPUT_LABELS = ("X[0]=-2", "X[1]=-0.5", "X[2]=0.5", "X[3]=2")
    OUTPUTS = ("Y[0]≈0.1192", "Y[1]≈0.3775", "Y[2]≈0.6225", "Y[3]≈0.8808")
    CODE = "x >= 0 ? 1/(1+expf(-x)) : expf(x)/(1+expf(x))"
    TEACHING_NOTE = "按输入符号选择代数等价分支，避免直接计算大幅正指数。"
