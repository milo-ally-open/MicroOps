"""Reusable, source-faithful visuals for the repository's elementwise CUDA v1 scenes."""

from manim import *


class ElementwiseV1Scene(Scene):
    OPERATOR = "operator"
    FORMULA = "Y[i] = f(X[i])"
    INPUTS = ()
    OUTPUTS = ()
    INPUT_LABELS = ()
    CODE = "Y[idx] = f(X[idx]);"
    TEACHING_NOTE = "Each valid index has one logical owner."

    def construct(self):
        title = Text(f"{self.OPERATOR} · CUDA v1", font_size=38, weight=BOLD)
        title.to_edge(UP, buff=0.28)
        subtitle = Text("逐元素并行：一个逻辑线程负责一个元素", font_size=23, color=GRAY_A)
        subtitle.next_to(title, DOWN, buff=0.12)
        formula = Text(self.FORMULA, font_size=27, color=YELLOW)
        formula.scale(min(1.0, 12.5 / formula.width))
        formula.move_to(UP * 1.55)

        self.play(Write(title), FadeIn(subtitle, shift=DOWN * 0.1))
        self.play(FadeIn(formula, shift=DOWN * 0.1))

        # Four enlarged lanes are a teaching sample, not a four-thread launch.
        lane_count = min(4, len(self.OUTPUTS))
        centers = [((i - (lane_count - 1) / 2) * 2.65) for i in range(lane_count)]
        cards = VGroup()
        for lane, x in enumerate(centers):
            card = RoundedRectangle(width=2.35, height=3.0, corner_radius=0.12)
            card.set_stroke(BLUE_C, 2).set_fill(BLUE_E, opacity=0.32).move_to([x, -0.2, 0])
            worker = Text(f"thread {lane}", font_size=19, color=BLUE_B).move_to(card.get_top() + DOWN * 0.35)
            index = Text(f"idx = {lane}", font_size=19).next_to(worker, DOWN, buff=0.22)
            input_text = Text(self.INPUT_LABELS[lane], font_size=18, color=TEAL_A)
            input_text.next_to(index, DOWN, buff=0.34)
            arrow = Arrow(input_text.get_bottom(), input_text.get_bottom() + DOWN * 0.32,
                          buff=0.04, stroke_width=2, color=GRAY_B)
            output_text = Text(self.OUTPUTS[lane], font_size=18, color=GREEN_A)
            output_text.next_to(arrow, DOWN, buff=0.12)
            card_group = VGroup(card, worker, index, input_text, arrow, output_text)
            cards.add(card_group)

        self.play(LaggedStart(*(FadeIn(card, shift=UP * 0.12) for card in cards), lag_ratio=0.12))
        note = Text(self.TEACHING_NOTE, font_size=19, color=GRAY_A)
        note.to_edge(DOWN, buff=0.28)
        self.play(Write(note))

        self.play(FadeOut(cards), FadeOut(note), FadeOut(formula))

        # Show a real tail guard from this repository's 256-thread CUDA launch.
        tail_title = Text("边界保护：最后一个 block 的尾部", font="SimSun", font_size=25, color=YELLOW)
        tail_title.move_to(UP * 1.5)
        tail_shape = Text("N = 258  →  grid = 2 blocks × 256 threads/block",
                          font="DejaVu Sans Mono", font_size=20, color=GRAY_A)
        tail_shape.next_to(tail_title, DOWN, buff=0.2)
        lane_cells = VGroup()
        for thread_idx in range(4):
            idx = 256 + thread_idx
            active = idx < 258
            cell = RoundedRectangle(width=2.15, height=1.38, corner_radius=0.1)
            cell.set_stroke(GREEN_C if active else GRAY_B, 2)
            cell.set_fill(GREEN_E if active else GRAY_E, opacity=0.45 if active else 0.18)
            cell.move_to([(thread_idx - 1.5) * 2.45, -0.2, 0])
            thread = Text(f"threadIdx={thread_idx}", font_size=16,
                          color=GREEN_A if active else GRAY_A).move_to(cell.get_top() + DOWN * 0.32)
            index = Text(f"idx={idx}", font_size=18).next_to(thread, DOWN, buff=0.16)
            state = Text("valid" if active else "masked", font_size=16,
                         color=GREEN_A if active else GRAY_B).next_to(index, DOWN, buff=0.12)
            lane_cells.add(VGroup(cell, thread, index, state))
        guard = Text("if (idx < N)", font="DejaVu Sans Mono", font_size=24, color=YELLOW_A)
        guard.move_to(DOWN * 1.35)
        omitted = Text("仅放大 threadIdx 0–3；block 中其余 252 个线程未画出。",
                       font="SimSun", font_size=17, color=GRAY_A)
        omitted.to_edge(DOWN, buff=0.2)
        self.play(Write(tail_title), FadeIn(tail_shape),
                  LaggedStart(*(FadeIn(cell, shift=UP * 0.1) for cell in lane_cells), lag_ratio=0.1))
        self.play(Write(guard), FadeIn(omitted))
        self.wait(1.5)
        self.play(FadeOut(tail_title), FadeOut(tail_shape), FadeOut(lane_cells),
                  FadeOut(guard), FadeOut(omitted))

        code = Text(self.CODE, font="DejaVu Sans Mono", font_size=23, color=YELLOW_A)
        code.scale(min(1.0, 12.5 / code.width)).move_to(DOWN * 0.1)
        # The source cue follows the ownership view; it is not a physical timeline.
        self.play(FadeIn(code, shift=UP * 0.12))
        self.wait(1.2)
        self.play(FadeOut(code))

        memory_title = Text("数据路径（概念示意）", font_size=29, color=YELLOW)
        memory_title.move_to(UP * 1.2)
        memory = VGroup(
            self._memory_box("GPU 全局内存", "输入向量"),
            Arrow(LEFT * 1.7, RIGHT * 0.2, color=BLUE_B, buff=0.08),
            self._memory_box("逻辑线程", "load → compute"),
            Arrow(LEFT * 0.2, RIGHT * 1.7, color=GREEN_B, buff=0.08),
            self._memory_box("GPU 全局内存", "输出向量"),
        ).arrange(RIGHT, buff=0.18).scale(0.88).move_to(DOWN * 0.1)
        mapping = Text("host 以 256 threads/block 计算 grid；示意中的线程并不代表同时执行顺序",
                       font_size=18, color=GRAY_A)
        mapping.to_edge(DOWN, buff=0.35)
        self.play(Write(memory_title), FadeIn(memory, shift=UP * 0.12), FadeIn(mapping))
        self.wait(2)

    @staticmethod
    def _memory_box(title, detail):
        box = RoundedRectangle(width=2.55, height=1.35, corner_radius=0.12)
        box.set_stroke(TEAL_C, 2).set_fill(BLACK, opacity=0.2)
        labels = VGroup(Text(title, font_size=19), Text(detail, font_size=16, color=GRAY_A))
        labels.arrange(DOWN, buff=0.14).move_to(box)
        return VGroup(box, labels)
