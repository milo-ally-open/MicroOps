"""Teaching scene based on src/cuda/kernel/sub_kernel_v1.cu."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "skills/ops-manim/scripts"))
from elementwise_scene import ElementwiseV1Scene  # noqa: E402


class SubV1Walkthrough(ElementwiseV1Scene):
    OPERATOR = "Subtraction"
    FORMULA = "C[i] = A[i] - B[i]"
    INPUT_LABELS = ("A[0]=5, B[0]=1", "A[1]=-2, B[1]=4", "A[2]=3.5, B[2]=-1.5", "A[3]=8, B[3]=2")
    OUTPUTS = ("C[0]=4", "C[1]=-6", "C[2]=5", "C[3]=6")
    CODE = "C[idx] = A[idx] - B[idx];"
    TEACHING_NOTE = "同一 idx 读取 A、B，并写入唯一的 C[idx]。"
