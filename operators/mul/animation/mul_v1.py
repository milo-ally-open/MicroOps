"""Teaching scene based on src/cuda/kernel/mul_kernel_v1.cu."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "skills/ops-manim/scripts"))
from elementwise_scene import ElementwiseV1Scene  # noqa: E402


class MulV1Walkthrough(ElementwiseV1Scene):
    OPERATOR = "Multiplication"
    FORMULA = "C[i] = A[i] × B[i]"
    INPUT_LABELS = ("A[0]=2, B[0]=3", "A[1]=-3, B[1]=2", "A[2]=0.5, B[2]=-8", "A[3]=4, B[3]=0.25")
    OUTPUTS = ("C[0]=6", "C[1]=-6", "C[2]=-4", "C[3]=1")
    CODE = "C[idx] = A[idx] * B[idx];"
    TEACHING_NOTE = "四个独立乘法各自消费一对同索引输入。"
