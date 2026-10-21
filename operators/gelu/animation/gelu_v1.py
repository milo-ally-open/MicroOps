"""Teaching scene based on src/cuda/kernel/gelu_kernel_v1.cu."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "skills/ops-manim/scripts"))
from elementwise_scene import ElementwiseV1Scene  # noqa: E402


class GeluV1Walkthrough(ElementwiseV1Scene):
    OPERATOR = "GELU"
    FORMULA = "0.5x(1+tanh(√(2/π)(x+0.044715x³)))"
    INPUT_LABELS = ("X[0]=-2", "X[1]=-0.5", "X[2]=0.5", "X[3]=2")
    OUTPUTS = ("Y[0]≈-0.0454", "Y[1]≈-0.1543", "Y[2]≈0.3457", "Y[3]≈1.9546")
    CODE = "0.5f*x*(1.0f+tanhf(0.79788456f*(x+0.044715f*x*x*x)))"
    TEACHING_NOTE = "展示的是 CUDA v1 使用的 tanh 近似 GELU 公式。"
