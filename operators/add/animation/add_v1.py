"""Teaching animation for the repository's CUDA add_kernel_v1."""

from manim import *


FONT_CJK = "Droid Sans Fallback"
FONT_CODE = "DejaVu Sans Mono"


class AddV1Walkthrough(Scene):
    """Explain v1's launch geometry, global index, guard, and one add."""

    def construct(self):
        self.camera.background_color = "#101722"

        # Objective: make the host's 256-thread launch and the kernel's
        # one-thread/one-element mapping concrete, including the tail guard.
        title = Text(
            "CUDA add_kernel_v1",
            font=FONT_CJK,
            font_size=42,
            color="#F2F6FC",
        ).to_edge(UP, buff=0.38)
        subtitle = Text(
            "一线程处理一个 float32 元素",
            font=FONT_CJK,
            font_size=27,
            color="#9EADC2",
        ).next_to(title, DOWN, buff=0.18)
        self.play(FadeIn(title, shift=DOWN * 0.15), FadeIn(subtitle))

        launch = Text(
            "示意输入 N = 258    |    每个 block 有 256 个线程    |    gridDim.x = 2",
            font=FONT_CJK,
            font_size=23,
            color="#5CC8FF",
        ).move_to([0, 1.75, 0])
        self.play(Write(launch))

        block0 = self._block_card(
            "blockIdx.x = 0",
            "threadIdx.x: 0 ... 255",
            "global idx: 0 ... 255  ✓",
            "前 256 个元素均有效",
            "#254D70",
        ).move_to([-3.25, 0.35, 0])
        block1 = self._block_card(
            "blockIdx.x = 1",
            "thread 0 → idx 256  ✓",
            "thread 1 → idx 257  ✓",
            "thread 2 → idx 258  ✕ 越界",
            "#315C53",
        ).move_to([3.25, 0.35, 0])
        self.play(FadeIn(block0, shift=LEFT * 0.2), FadeIn(block1, shift=RIGHT * 0.2))
        self.wait(0.8)

        mapping = Text(
            "idx = blockIdx.x * blockDim.x + threadIdx.x",
            font=FONT_CODE,
            font_size=27,
            color="#F2F6FC",
        ).move_to([0, -1.35, 0])
        example = Text(
            "例：1 * 256 + 1 = 257",
            font=FONT_CJK,
            font_size=27,
            color="#5CC8FF",
        ).next_to(mapping, DOWN, buff=0.22)
        self.play(FadeOut(block0), FadeOut(block1), FadeIn(mapping, shift=UP * 0.15))
        self.play(Write(example))

        guard_true = Text(
            "idx < N  →  257 < 258  →  通过边界检查",
            font=FONT_CJK,
            font_size=27,
            color="#55D6A0",
        ).move_to([0, -2.45, 0])
        self.play(Transform(example, guard_true))

        tail = Text(
            "thread 2 得到 idx=258：检查失败，因此不读写数组",
            font=FONT_CJK,
            font_size=23,
            color="#FFBE55",
        ).next_to(example, DOWN, buff=0.22)
        self.play(FadeIn(tail, shift=UP * 0.12))
        self.wait(0.8)

        flow_title = Text(
            "一个有效线程的读—算—写",
            font=FONT_CJK,
            font_size=26,
            color="#F2F6FC",
        ).move_to([0, 1.75, 0])
        a_value = self._value_card("A[257]", "3.0", "#5CC8FF").move_to([-4.5, 0.15, 0])
        b_value = self._value_card("B[257]", "4.0", "#5CC8FF").move_to([-4.5, -1.25, 0])
        adder = Circle(
            radius=0.58,
            stroke_color="#FFBE55",
            stroke_width=2,
            fill_color="#302819",
            fill_opacity=1,
        ).move_to([0, -0.55, 0])
        plus = Text("+", font=FONT_CODE, font_size=42, color="#FFBE55").move_to(adder)
        c_value = self._value_card("C[257]", "7.0", "#55D6A0").move_to([4.2, -0.55, 0])
        arrow_a = Arrow(a_value.get_right(), adder.get_left() + UP * 0.2, buff=0.12, color="#5CC8FF")
        arrow_b = Arrow(b_value.get_right(), adder.get_left() + DOWN * 0.2, buff=0.12, color="#5CC8FF")
        arrow_c = Arrow(adder.get_right(), c_value.get_left(), buff=0.12, color="#55D6A0")
        equation = Text(
            "C[257] = A[257] + B[257]",
            font=FONT_CODE,
            font_size=24,
            color="#C5D2E2",
        ).move_to([0, -2.35, 0])
        self.play(
            FadeOut(tail),
            FadeOut(mapping),
            FadeOut(example),
            FadeOut(launch),
            FadeIn(flow_title),
        )
        self.play(
            Create(a_value),
            Create(b_value),
            Create(adder),
            FadeIn(plus),
            Create(c_value),
        )
        self.play(GrowArrow(arrow_a), GrowArrow(arrow_b))
        self.play(GrowArrow(arrow_c), Write(equation))

        takeaway = Text(
            "每个有效 idx 独立完成一次加法并写回 C[idx]",
            font=FONT_CJK,
            font_size=24,
            color="#55D6A0",
        ).move_to([0, -3.15, 0])
        self.play(Write(takeaway))
        self.wait(1.5)

    @staticmethod
    def _block_card(title, line1, line2, line3, color):
        panel = RoundedRectangle(
            width=5.9,
            height=2.35,
            corner_radius=0.14,
            stroke_color=color,
            stroke_width=2,
            fill_color="#172333",
            fill_opacity=1,
        )
        content = VGroup(
            Text(title, font=FONT_CODE, font_size=23, color="#F2F6FC"),
            Text(line1, font=FONT_CODE, font_size=17, color="#C5D2E2"),
            Text(line2, font=FONT_CODE, font_size=17, color="#C5D2E2"),
            Text(line3, font=FONT_CJK, font_size=19, color=color),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.17)
        content.move_to(panel.get_center())
        return VGroup(panel, content)

    @staticmethod
    def _value_card(name, value, color):
        panel = RoundedRectangle(
            width=2.55,
            height=1.05,
            corner_radius=0.12,
            stroke_color=color,
            stroke_width=2,
            fill_color="#172333",
            fill_opacity=1,
        )
        content = VGroup(
            Text(name, font=FONT_CODE, font_size=19, color="#C5D2E2"),
            Text(value, font=FONT_CODE, font_size=27, color=color),
        ).arrange(DOWN, buff=0.04)
        content.move_to(panel.get_center())
        return VGroup(panel, content)
