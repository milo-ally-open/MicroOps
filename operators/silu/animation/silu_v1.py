"""Teaching scene based on src/cuda/kernel/silu_kernel_v1.cu."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "skills/ops-manim/scripts"))
from elementwise_scene import ElementwiseV1Scene  # noqa: E402


class SiluV1Walkthrough(ElementwiseV1Scene):
    OPERATOR = "SiLU"
    FORMULA = "Y[i] = X[i] / (1 + exp(-X[i]))"
    INPUT_LABELS = ("X[0]=-2", "X[1]=-0.5", "X[2]=0.5", "X[3]=2")
    OUTPUTS = ("Y[0]≈-0.2384", "Y[1]≈-0.1888", "Y[2]≈0.3112", "Y[3]≈1.7616")
    CODE = "Y[idx] = x / (1.0f + expf(-x));"
    TEACHING_NOTE = "SiLU 用 sigmoid 门控输入；每个线程只处理自己的 x。"
