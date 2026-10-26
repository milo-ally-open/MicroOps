"""Hardware-aware, story-led Manim scenes for the repository's CUDA v1 operators."""

from __future__ import annotations

import math

import numpy as np
from manim import *


# Visual grammar: data keeps its color as it moves; amber marks a decision or
# arithmetic event; green marks a produced value. Shapes remain schematic.
BG = "#07111F"
GRID = "#12243A"
PANEL = "#0D1C2E"
WHITE = "#F4F7FB"
MUTED = "#A5B4C8"
BLUE = "#55C7F5"
VIOLET = "#A78BFA"
GREEN = "#54E0A0"
AMBER = "#FFC857"
RED = "#FF7185"
FONT_CJK = "SimSun"
FONT_CODE = "DejaVu Sans Mono"


class ElementwiseV1Scene(Scene):
    """Tell a short mathematical story, then map it to the actual CUDA v1."""

    OPERATOR = "Elementwise"
    OP_KEY = "add"
    FORMULA = "Y[i] = f(X[i])"
    INPUT_LABELS: tuple[str, ...] = ()
    OUTPUTS: tuple[str, ...] = ()
    CODE = "Y[idx] = f(X[idx]);"
    GRAPH_KEY: str | None = None
    GRAPH_Y_RANGE = (-2.0, 2.0)
    GRAPH_NOTE = ""

    GRAPH_FUNCTIONS = {
        "relu": lambda x: max(0.0, x),
        "gelu": lambda x: 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3))),
        "silu": lambda x: x / (1.0 + math.exp(-x)),
        "sigmoid": lambda x: 1.0 / (1.0 + math.exp(-x)),
        "tanh": math.tanh,
    }

    def construct(self):
        self.camera.background_color = BG
        self.add(self.background_grid())
        self.opening_question()
        if self.GRAPH_KEY:
            self.activation_story()
        else:
            self.binary_story()
        self.cuda_story()
        self.optimization_story()
        if getattr(self, "FRAMEWORK_OPTIMIZATION", None) == "relu_fusion":
            self.relu_fusion_story()
        self.ending()

    def background_grid(self):
        lines = VGroup()
        for x in np.arange(-7, 7.1, 1):
            lines.add(Line([x, -4.2, 0], [x, 4.2, 0], color=GRID, stroke_width=0.5))
        for y in np.arange(-4, 4.1, 1):
            lines.add(Line([-7.5, y, 0], [7.5, y, 0], color=GRID, stroke_width=0.5))
        return lines.set_opacity(0.58)

    def opening_question(self):
        title = Text(self.OPERATOR, font=FONT_CJK, font_size=44, color=WHITE)
        title.move_to(UP * 1.05)
        formula = Text(self.FORMULA, font=FONT_CODE, font_size=31, color=AMBER)
        formula.scale(min(1, 11.8 / formula.width)).next_to(title, DOWN, buff=0.42)
        subtitle = Text("把一个公式，变成一群线程的工作", font=FONT_CJK, font_size=23, color=MUTED)
        subtitle.next_to(formula, DOWN, buff=0.58)
        badge = self.badge("CUDA v1 · float32 · 一线程一个元素", BLUE)
        badge.to_edge(DOWN, buff=0.48)
        self.play(FadeIn(title, shift=UP * 0.18), run_time=0.7)
        self.play(Write(formula), run_time=0.9)
        self.play(FadeIn(subtitle), FadeIn(badge), run_time=0.6)
        self.wait(1.1)
        self.play(FadeOut(VGroup(title, formula, subtitle, badge)), run_time=0.55)

    def binary_story(self):
        """A distinct visual metaphor for each arithmetic relation."""
        if self.OP_KEY == "add":
            self.addition_story()
        elif self.OP_KEY == "sub":
            self.subtraction_story()
        elif self.OP_KEY == "mul":
            self.multiplication_story()
        else:
            self.division_story()

    def addition_story(self):
        title = self.header("加法：同一位置的两份贡献，汇成一个结果")
        columns = VGroup()
        pairs = (("1", "8", "9"), ("4", "2", "6"), ("2", "9", "11"), ("10", "20", "30"))
        for i, (a, b, c) in enumerate(pairs):
            box = RoundedRectangle(width=2.35, height=3.55, corner_radius=0.14,
                                   stroke_color=BLUE, stroke_width=1.6, fill_color=PANEL, fill_opacity=0.92)
            idx = Text(f"index {i}", font=FONT_CODE, font_size=15, color=MUTED).move_to(box.get_top() + DOWN * 0.35)
            av = Text(f"A[{i}] = {a}", font=FONT_CODE, font_size=17, color=BLUE)
            bv = Text(f"B[{i}] = {b}", font=FONT_CODE, font_size=17, color=VIOLET)
            cv = Text(f"C[{i}] = {c}", font=FONT_CODE, font_size=17, color=GREEN)
            plus = Text("+", font=FONT_CODE, font_size=25, color=AMBER)
            equal = Text("=", font=FONT_CODE, font_size=25, color=AMBER)
            content = VGroup(idx, av, plus, bv, equal, cv).arrange(DOWN, buff=0.17).move_to(box)
            columns.add(VGroup(box, content))
        columns.arrange(RIGHT, buff=0.17).move_to(UP * 0.03)
        note = Text("每一列独立：A[i] 与 B[i] 不会跨下标混合。", font=FONT_CJK, font_size=21, color=MUTED).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(title))
        self.play(LaggedStart(*[FadeIn(column, shift=UP * 0.18) for column in columns], lag_ratio=0.14), run_time=1.0)
        self.play(*[Indicate(column[1][-1], color=GREEN, scale_factor=1.12) for column in columns], FadeIn(note), run_time=0.8)
        self.wait(1.0)
        self.play(FadeOut(VGroup(title, columns, note)), run_time=0.55)

    def subtraction_story(self):
        title = self.header("减法：减去 B，就是沿数轴反向走 B")
        axis = NumberLine(x_range=[-7, 7, 1], length=10.2, include_numbers=False,
                          color=MUTED, include_tip=True, numbers_to_exclude=[-7, 7])
        axis.move_to(UP * 0.35)
        zero_x, a_x, result_x = axis.n2p(0), axis.n2p(3), axis.n2p(1)
        origin = Dot(zero_x, color=MUTED, radius=0.08)
        a_point = Dot(a_x, color=BLUE, radius=0.11)
        result = Dot(result_x, color=GREEN, radius=0.11)
        a_arrow = Arrow(zero_x, a_x, color=BLUE, buff=0, stroke_width=5)
        reverse_b = Arrow(a_x, result_x, color=AMBER, buff=0, stroke_width=5)
        a_label = Text("A = 3", font=FONT_CODE, font_size=22, color=BLUE).next_to(a_point, UP, buff=0.25)
        b_label = Text("减去 B = 2：从 3 向左走两格", font=FONT_CJK, font_size=18, color=AMBER).move_to(DOWN * 0.95)
        c_label = Text("A − B = 1", font=FONT_CODE, font_size=25, color=GREEN).next_to(result, DOWN, buff=0.3)
        identity = Text("A − B = A + (−B)", font=FONT_CODE, font_size=29, color=AMBER).move_to(DOWN * 1.75)
        vals = self.sample_strip(("5−1", "−2−4", "3.5−(−1.5)", "8−2"), ("4", "−6", "5", "6"))
        vals.to_edge(DOWN, buff=0.34)
        self.play(FadeIn(title), Create(axis), run_time=0.8)
        self.play(GrowFromCenter(origin), GrowArrow(a_arrow), GrowFromCenter(a_point), FadeIn(a_label), run_time=0.75)
        self.play(GrowArrow(reverse_b), FadeIn(b_label), run_time=0.8)
        self.play(GrowFromCenter(result), FadeIn(c_label), run_time=0.5)
        self.play(Write(identity), FadeIn(vals), run_time=0.7)
        self.wait(1.0)
        self.play(FadeOut(VGroup(title, axis, origin, a_point, result, a_arrow, reverse_b, a_label, b_label, c_label, identity, vals)), run_time=0.55)

    def multiplication_story(self):
        title = self.header("乘法：每个位置独立缩放，并由符号决定方向")
        plane = Axes(x_range=[-4, 4, 1], y_range=[-4, 4, 1], x_length=5.8, y_length=4.1,
                     axis_config={"color": MUTED, "include_tip": True}).move_to(LEFT * 3.1 + DOWN * 0.25)
        labels = VGroup(Text("A", font=FONT_CODE, font_size=20, color=BLUE).next_to(plane.x_axis, RIGHT),
                        Text("B", font=FONT_CODE, font_size=20, color=VIOLET).next_to(plane.y_axis, UP))
        samples = [(2, 3, 6), (-3, 2, -6), (0.5, -3, -1.5), (4, 0.25, 1)]
        dots = VGroup(*[Dot(plane.c2p(a, b), color=AMBER, radius=0.075) for a, b, _ in samples])
        ring = Circle(radius=0.22, color=GREEN, stroke_width=3).move_to(RIGHT * 3.1 + UP * 0.35)
        times = Text("×", font=FONT_CODE, font_size=43, color=AMBER).move_to(RIGHT * 0.75 + UP * 0.35)
        results = self.sample_strip(tuple(f"{a}×{b}" for a, b, _ in samples), tuple(str(c) for _, _, c in samples))
        results.to_edge(DOWN, buff=0.3)
        explanation = Text("点的位置表示输入对 (A[i], B[i])；右侧结果仍按同一 i 写回。", font=FONT_CJK, font_size=18, color=MUTED)
        explanation.move_to(UP * 2.0 + RIGHT * 2.5).scale(0.9)
        sign = Text("同号 → 正    异号 → 负", font=FONT_CJK, font_size=22, color=GREEN).move_to(UP * 1.0 + RIGHT * 2.5)
        self.play(FadeIn(title), Create(plane), FadeIn(labels), run_time=0.8)
        self.play(LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.15), run_time=0.8)
        self.play(FadeIn(times), Create(ring), FadeIn(explanation), FadeIn(sign), run_time=0.7)
        self.play(FadeIn(results, shift=UP * 0.15), run_time=0.65)
        self.wait(0.9)
        self.play(FadeOut(VGroup(title, plane, labels, dots, ring, times, results, explanation, sign)), run_time=0.55)

    def division_story(self):
        title = self.header("除法：回答‘分母乘几，才能得到分子？’")
        eq = VGroup(Text("8", font=FONT_CODE, font_size=58, color=BLUE),
                    Text("÷", font=FONT_CODE, font_size=48, color=AMBER),
                    Text("2", font=FONT_CODE, font_size=58, color=VIOLET),
                    Text("=", font=FONT_CODE, font_size=48, color=AMBER),
                    Text("4", font=FONT_CODE, font_size=64, color=GREEN)).arrange(RIGHT, buff=0.42)
        eq.move_to(UP * 0.65)
        bars = VGroup()
        for i in range(4):
            cell = RoundedRectangle(width=1.15, height=0.66, corner_radius=0.08,
                                    stroke_color=VIOLET, fill_color=VIOLET, fill_opacity=0.13)
            cell.move_to(LEFT * 3.25 + RIGHT * i * 0.86 + DOWN * 0.65)
            bars.add(cell)
        bracket = Brace(bars, DOWN, color=AMBER)
        chunk = Text("每份 2，一共 4 份", font=FONT_CJK, font_size=21, color=AMBER).next_to(bracket, DOWN, buff=0.1)
        label = Text("4 × 2 = 8", font=FONT_CODE, font_size=23, color=GREEN).move_to(RIGHT * 3.0 + DOWN * 0.65)
        vals = self.sample_strip(("8÷(−2)", "3÷2", "−9÷(−3)", "1÷4"), ("−4", "1.5", "3", "0.25"))
        vals.to_edge(DOWN, buff=0.3)
        caveat = Text("演示选用非零分母；零除与 NaN/Inf 遵循设备浮点语义。", font=FONT_CJK, font_size=17, color=MUTED).to_edge(UP, buff=0.65)
        self.play(FadeIn(title), FadeIn(caveat), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(m, shift=UP * 0.16) for m in eq], lag_ratio=0.1), run_time=1.0)
        self.play(LaggedStart(*[Create(b) for b in bars], lag_ratio=0.09), GrowFromCenter(bracket), FadeIn(chunk), FadeIn(label), run_time=0.9)
        self.play(FadeIn(vals), run_time=0.6)
        self.wait(0.9)
        self.play(FadeOut(VGroup(title, caveat, eq, bars, bracket, chunk, label, vals)), run_time=0.55)

    def activation_story(self):
        title = self.header(self.activation_title())
        graph = self.make_graph()
        curve = graph["curve"]
        axes = graph["axes"]
        feature = Text(self.activation_caption(), font=FONT_CJK, font_size=18, color=AMBER)
        feature.to_edge(DOWN, buff=0.32)
        markers, drops = self.graph_markers(axes)
        self.play(FadeIn(title), Create(axes), FadeIn(graph["ticks"]), run_time=0.8)
        self.play(Create(curve, run_time=1.7), run_time=1.7)
        self.play(LaggedStart(*[Create(d) for d in drops], lag_ratio=0.08),
                  LaggedStart(*[GrowFromCenter(m) for m in markers], lag_ratio=0.08), run_time=0.9)
        self.play(FadeIn(feature), run_time=0.45)
        self.wait(1.2)
        self.play(FadeOut(VGroup(title, axes, graph["ticks"], curve, markers, drops, feature)), run_time=0.55)
        self.activation_detail_story()

    def make_graph(self):
        y_min, y_max = self.GRAPH_Y_RANGE
        axes = Axes(x_range=[-4, 4, 2], y_range=[y_min, y_max, (y_max-y_min)/4],
                    x_length=7.3, y_length=4.55,
                    axis_config={"color": MUTED, "stroke_width": 2, "include_tip": True}).move_to(LEFT * 1.65 + DOWN * 0.12)
        ticks = VGroup()
        for x in (-4, -2, 0, 2, 4):
            ticks.add(Text(str(x), font=FONT_CODE, font_size=13, color=MUTED).move_to(axes.c2p(x, y_min) + DOWN * 0.18))
        for y in np.linspace(y_min, y_max, 5):
            ticks.add(Text(f"{y:g}", font=FONT_CODE, font_size=12, color=MUTED).move_to(axes.c2p(-4, y) + LEFT * 0.27))
        grid = VGroup(*[Line(axes.c2p(x, y_min), axes.c2p(x, y_max), color=GRID, stroke_width=1) for x in (-4, -2, 0, 2, 4)],
                      *[Line(axes.c2p(-4, y), axes.c2p(4, y), color=GRID, stroke_width=1) for y in np.linspace(y_min,y_max,5)])
        axes.add(grid)
        xlab = Text("输入 x", font=FONT_CJK, font_size=16, color=MUTED).next_to(axes.x_axis, DOWN, buff=0.3)
        ylab = Text("输出 y", font=FONT_CJK, font_size=16, color=MUTED).next_to(axes.y_axis, LEFT, buff=0.22).rotate(PI/2)
        axes.add(xlab, ylab)
        curve = axes.plot(self.GRAPH_FUNCTIONS[self.GRAPH_KEY], x_range=[-4,4], color=GREEN, stroke_width=4)
        formula = Text(self.FORMULA, font=FONT_CODE, font_size=18, color=BLUE)
        formula.scale(min(1, 3.9 / formula.width)).move_to(RIGHT * 4.75 + UP * 1.45)
        samples = self.sample_strip(self.INPUT_LABELS, self.OUTPUTS, compact=True)
        samples.scale(0.66).move_to(RIGHT * 4.7 + DOWN * 0.45)
        return {"axes": axes, "ticks": VGroup(ticks, formula, samples), "curve": curve}

    def graph_markers(self, axes):
        markers, drops = VGroup(), VGroup()
        for x in (-2.0, -0.5, 0.5, 2.0):
            y = self.GRAPH_FUNCTIONS[self.GRAPH_KEY](x)
            p = axes.c2p(x, y)
            markers.add(Dot(p, color=AMBER, radius=0.07))
            drops.add(DashedLine(axes.c2p(x, 0), p, color=AMBER, dash_length=0.06, stroke_width=1.3))
        return markers, drops

    def activation_detail_story(self):
        """Give each activation a full-frame explanation beat after its curve."""
        title = self.header({
            "relu": "ReLU 的折点：负值归零，正值通过",
            "gelu": "GELU v1：把输入送过平滑近似门",
            "silu": "SiLU：信号乘上一个连续门值",
            "sigmoid": "Sigmoid：按输入符号选择稳定等价式",
            "tanh": "tanh：中心对称，远离零后逐渐饱和",
        }[self.GRAPH_KEY])
        self.play(FadeIn(title), run_time=0.45)
        if self.GRAPH_KEY == "relu":
            self.relu_hinge_beat()
        elif self.GRAPH_KEY == "gelu":
            self.gelu_gate_beat()
        elif self.GRAPH_KEY == "silu":
            self.silu_gate_beat()
        elif self.GRAPH_KEY == "sigmoid":
            self.sigmoid_stability_beat()
        else:
            self.tanh_symmetry_beat()
        self.play(FadeOut(title), run_time=0.4)

    def relu_hinge_beat(self):
        cases = VGroup()
        for x, y, color in (("−2", "0", RED), ("+2", "+2", GREEN)):
            left = self.pill(f"x = {x}", BLUE)
            arrow = Arrow(ORIGIN, RIGHT * 1.0, color=AMBER, buff=0.04, stroke_width=3)
            result = self.pill(f"y = {y}", color)
            cases.add(VGroup(left, arrow, result).arrange(RIGHT, buff=0.45))
        cases.arrange(DOWN, buff=0.65).move_to(UP * 0.3)
        rule = Text("if x > 0: y = x     else: y = 0", font=FONT_CODE, font_size=27, color=AMBER).move_to(DOWN * 1.6)
        self.play(LaggedStart(*[FadeIn(case, shift=RIGHT*0.16) for case in cases], lag_ratio=0.25), run_time=1.0)
        self.play(Write(rule), run_time=0.65)
        self.wait(0.6)
        self.play(FadeOut(VGroup(cases, rule)), run_time=0.35)

    def gelu_gate_beat(self):
        # These stages reproduce the tanh approximation in gelu_kernel_v1.cu.
        stages = [
            ("u = 0.79788456 · (x + 0.044715x³)", BLUE),
            ("g = tanh(u)", VIOLET),
            ("y = 0.5x · (1 + g)", GREEN),
        ]
        rows = VGroup()
        for number, (formula, color) in enumerate(stages, 1):
            badge = Circle(radius=0.2, color=color, stroke_width=2)
            badge.add(Text(str(number), font=FONT_CODE, font_size=15, color=color).move_to(badge))
            text = Text(formula, font=FONT_CODE, font_size=23, color=WHITE)
            text.scale(min(1, 10.8/text.width))
            rows.add(VGroup(badge, text).arrange(RIGHT, buff=0.3))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.5).move_to(UP * 0.15)
        arrows = VGroup(*[Arrow(rows[i].get_bottom(), rows[i+1].get_top(), color=AMBER, buff=0.08, stroke_width=2)
                          for i in range(2)])
        note = Text("这条曲线使用 tanh 近似，不是精确 erf 公式。", font=FONT_CJK, font_size=19, color=AMBER).move_to(DOWN * 2.0)
        self.play(FadeIn(rows[0], shift=DOWN*0.12), run_time=0.55)
        self.play(GrowArrow(arrows[0]), FadeIn(rows[1], shift=DOWN*0.12), run_time=0.65)
        self.play(GrowArrow(arrows[1]), FadeIn(rows[2], shift=DOWN*0.12), run_time=0.65)
        self.play(FadeIn(note), run_time=0.4)
        self.wait(0.6)
        self.play(FadeOut(VGroup(rows, arrows, note)), run_time=0.35)

    def silu_gate_beat(self):
        signal = self.pill("信号  x = −2", BLUE)
        gate = self.pill("门  σ(x) ≈ 0.1192", VIOLET)
        product = self.pill("y = x · σ(x) ≈ −0.2384", GREEN)
        inputs = VGroup(signal, gate).arrange(DOWN, buff=0.5).move_to(LEFT * 3.0 + UP * 0.3)
        result = product.move_to(RIGHT * 3.0 + UP * 0.3)
        multiply = Circle(radius=0.37, color=AMBER, stroke_width=2).move_to(ORIGIN + UP * 0.3)
        multiply.add(Text("×", font=FONT_CODE, font_size=35, color=AMBER).move_to(multiply))
        arrows = VGroup(Arrow(inputs[0].get_right(), multiply.get_left(), color=BLUE, buff=0.08),
                        Arrow(inputs[1].get_right(), multiply.get_left(), color=VIOLET, buff=0.08),
                        Arrow(multiply.get_right(), result.get_left(), color=GREEN, buff=0.08))
        note = Text("门值连续地调节信号：它不是开 / 关开关。", font=FONT_CJK, font_size=21, color=WHITE).move_to(DOWN * 1.35)
        self.play(FadeIn(inputs[0], shift=RIGHT*0.15), FadeIn(inputs[1], shift=RIGHT*0.15), run_time=0.7)
        self.play(GrowFromCenter(multiply), GrowArrow(arrows[0]), GrowArrow(arrows[1]), run_time=0.65)
        self.play(GrowArrow(arrows[2]), FadeIn(result, shift=RIGHT*0.15), Write(note), run_time=0.75)
        self.wait(0.6)
        self.play(FadeOut(VGroup(inputs, result, multiply, arrows, note)), run_time=0.35)

    def sigmoid_stability_beat(self):
        split = self.pill("检查 x 的符号", AMBER).move_to(UP * 1.45)
        left = self.hardware_card("x ≥ 0", ["y = 1 / (1 + exp(−x))"], GREEN, width=5.1)
        right = self.hardware_card("x < 0", ["y = exp(x) / (1 + exp(x))"], VIOLET, width=5.1)
        branches = VGroup(left, right).arrange(RIGHT, buff=0.65).move_to(UP * 0.0)
        merge = self.pill("两式等价，输出都在 (0, 1)", BLUE).move_to(DOWN * 1.35)
        note = Text("负分支避免直接计算可能很大的 exp(−x)。", font=FONT_CJK, font_size=19, color=MUTED).move_to(DOWN * 2.15)
        self.play(FadeIn(split), run_time=0.35)
        self.play(FadeIn(branches[0], shift=DOWN*0.12), FadeIn(branches[1], shift=DOWN*0.12), run_time=0.7)
        self.play(FadeIn(merge), FadeIn(note), run_time=0.55)
        self.wait(0.6)
        self.play(FadeOut(VGroup(split, branches, merge, note)), run_time=0.35)

    def tanh_symmetry_beat(self):
        positive = self.pill("x = +2  →  tanh(x) ≈ +0.964", BLUE)
        negative = self.pill("x = −2  →  tanh(x) ≈ −0.964", VIOLET)
        examples = VGroup(positive, negative).arrange(DOWN, buff=0.55).move_to(UP * 0.35)
        law = Text("tanh(−x) = −tanh(x)", font=FONT_CODE, font_size=31, color=AMBER).move_to(DOWN * 1.2)
        limits = Text("输入绝对值继续增大时，输出趋近 −1 或 +1。", font=FONT_CJK, font_size=20, color=GREEN).move_to(DOWN * 2.0)
        self.play(FadeIn(positive, shift=RIGHT*0.16), run_time=0.5)
        self.play(FadeIn(negative, shift=LEFT*0.16), run_time=0.5)
        self.play(Write(law), FadeIn(limits), run_time=0.65)
        self.wait(0.6)
        self.play(FadeOut(VGroup(examples, law, limits)), run_time=0.35)

    def cuda_story(self):
        title = self.header("CUDA：把索引分给线程，再让线程读、算、写")
        launch = self.pill("N = 258（示例）     threads/block = 256     blocks = ceil(N/256) = 2", BLUE)
        launch.scale(0.84).move_to(UP * 2.3)
        self.play(FadeIn(title), FadeIn(launch), run_time=0.55)
        grid_cards = VGroup(self.hardware_card("Grid block 0", ["thread 0 … 255", "global idx 0 … 255"], BLUE),
                            self.hardware_card("Grid block 1", ["thread 0 … 255", "global idx 256 … 511"], VIOLET))
        grid_cards.arrange(RIGHT, buff=0.7).move_to(UP * 0.85)
        grid_label = Text("一个 block 被放到某个 SM 执行；具体落点由调度器决定。", font=FONT_CJK, font_size=17, color=MUTED).move_to(DOWN * 0.42)
        self.play(FadeIn(grid_cards[0], shift=UP*0.12), FadeIn(grid_cards[1], shift=UP*0.12), FadeIn(grid_label), run_time=0.75)
        self.wait(0.4)
        self.play(FadeOut(VGroup(grid_cards, grid_label)), run_time=0.35)

        sample_lanes = VGroup()
        lane_colors = (BLUE, VIOLET, AMBER, GREEN)
        for idx, (inputs, output) in enumerate(zip(self.INPUT_LABELS, self.OUTPUTS)):
            sample_lanes.add(self.hardware_card(
                f"threadIdx {idx} → idx {idx}", [inputs, output], lane_colors[idx], width=2.9
            ))
        sample_lanes.arrange(RIGHT, buff=0.16).move_to(UP * 0.9)
        ownership = Text("每个线程读取自己的下标，并写回同一个输出下标。", font=FONT_CJK, font_size=17, color=WHITE).move_to(DOWN * 0.3)
        path = VGroup(*[self.hardware_card(label, lines, color, width=3.2) for label, lines, color in (
            ("Global memory", ["load input value(s)"], BLUE),
            ("CUDA thread", ["apply this operator"], AMBER),
            ("Global memory", ["store output value"], GREEN),
        )]).arrange(RIGHT, buff=0.5).move_to(DOWN * 1.55)
        arrows = VGroup(Arrow(path[0].get_right(), path[1].get_left(), color=BLUE, buff=0.06),
                        Arrow(path[1].get_right(), path[2].get_left(), color=GREEN, buff=0.06))
        code = Text("idx = blockIdx.x * blockDim.x + threadIdx.x", font=FONT_CODE, font_size=18, color=AMBER)
        code.scale(min(1, 11.8/code.width)).move_to(DOWN * 2.85)
        flow_note = Text("示意数据依赖，不表示固定调度顺序、同时运行或实际性能。", font=FONT_CJK, font_size=14, color=MUTED).to_edge(DOWN, buff=0.12)
        self.play(LaggedStart(*[FadeIn(lane, shift=UP*0.1) for lane in sample_lanes], lag_ratio=0.08), FadeIn(ownership), run_time=0.85)
        self.play(FadeIn(path[0]), GrowArrow(arrows[0]), FadeIn(path[1]), GrowArrow(arrows[1]), FadeIn(path[2]), Write(code), FadeIn(flow_note), run_time=0.9)
        self.wait(1.0)
        self.play(FadeOut(VGroup(sample_lanes, ownership, path, arrows, code, flow_note)), run_time=0.4)

        lanes = VGroup()
        for idx in (256, 257, 258, 259):
            valid = idx < 258
            color = GREEN if valid else RED
            rect = RoundedRectangle(width=2.55, height=0.9, corner_radius=0.12,
                                    stroke_color=color, fill_color=PANEL, fill_opacity=0.95)
            label = VGroup(Text(f"threadIdx {idx-256}", font=FONT_CODE, font_size=14, color=BLUE),
                           Text(f"idx {idx}  ·  {'active' if valid else 'guarded'}", font=FONT_CODE, font_size=15, color=color)).arrange(DOWN, buff=0.11).move_to(rect)
            lanes.add(VGroup(rect, label))
        lanes.arrange(RIGHT, buff=0.18).move_to(UP * 0.85)
        tail_code = Text("if (idx < N)  →  load · compute · store", font=FONT_CODE, font_size=20, color=AMBER)
        tail_code.scale(min(1, 11.8/tail_code.width)).move_to(DOWN * 0.15)
        tail_note = Text("N=258：idx 256、257 有效；258、259 被边界条件拦下。", font=FONT_CJK, font_size=17, color=MUTED).to_edge(DOWN, buff=0.22)
        self.play(LaggedStart(*[FadeIn(lane, shift=UP*0.1) for lane in lanes], lag_ratio=0.08), run_time=0.8)
        self.play(Write(tail_code), FadeIn(tail_note), run_time=0.65)
        self.wait(1.1)
        self.play(FadeOut(VGroup(title, launch, lanes, tail_code, tail_note)), run_time=0.55)

    def optimization_story(self):
        """Explain the v2 work granularity without claiming it always wins."""
        title = self.header("优化候选：改变每个线程的工作颗粒度")
        subtitle = Text("v1 与 v2 输出相同；关键是如何分配连续元素。", font=FONT_CJK,
                        font_size=21, color=WHITE).move_to(UP * 2.18)
        if self.GRAPH_KEY:
            lanes = VGroup()
            for thread in range(4):
                rect = RoundedRectangle(width=5.2, height=0.62, corner_radius=0.1,
                                        stroke_color=(BLUE, VIOLET, AMBER, GREEN)[thread],
                                        fill_color=PANEL, fill_opacity=0.9)
                label = Text(f"thread {thread}  →  idx {thread}, {4+thread}, {8+thread}, {12+thread}",
                             font=FONT_CODE, font_size=17, color=WHITE).move_to(rect)
                lanes.add(VGroup(rect, label))
            lanes.arrange(DOWN, buff=0.14).move_to(UP * 0.05)
            mapping_note = Text("blockDim=4 的 toy block：v2 展开 4 个标量；每段内相邻 lane 访问连续地址。",
                                font=FONT_CJK, font_size=16, color=MUTED).move_to(DOWN * 1.88)
            self.play(FadeIn(title), FadeIn(subtitle), run_time=0.55)
            self.play(LaggedStart(*[FadeIn(lane, shift=RIGHT * 0.12) for lane in lanes],
                                  lag_ratio=0.12), run_time=0.9)
            self.play(FadeIn(mapping_note), run_time=0.45)
            cards = VGroup(
                Text("每个元素仍读 X、写 Y：约 8 B/element。", font=FONT_CJK, font_size=16, color=BLUE),
                Text("展开可能降低索引/循环开销，但不会减少数据流量。", font=FONT_CJK, font_size=16, color=AMBER),
                Text("本轮测量没有证明 v2 对所有激活函数都更快。", font=FONT_CJK, font_size=16, color=GREEN),
            ).arrange(DOWN, buff=0.13).move_to(DOWN * 2.83)
            self.play(LaggedStart(*[FadeIn(card, shift=UP * 0.08) for card in cards],
                                  lag_ratio=0.16), run_time=0.8)
            self.wait(0.9)
            self.play(FadeOut(VGroup(title, subtitle, lanes, mapping_note, cards)), run_time=0.45)
            return

        title_line = Text("对齐时，一条 float4 指令路径覆盖四个连续元素", font=FONT_CJK,
                          font_size=20, color=WHITE).move_to(UP * 2.05)
        vectors = VGroup()
        for name, values, color in (("A", ("a₀", "a₁", "a₂", "a₃"), BLUE),
                                    ("B", ("b₀", "b₁", "b₂", "b₃"), VIOLET),
                                    ("C", ("c₀", "c₁", "c₂", "c₃"), GREEN)):
            cells = VGroup(Text(name, font=FONT_CODE, font_size=20, color=color))
            for value in values:
                box = RoundedRectangle(width=1.15, height=0.58, corner_radius=0.08,
                                       stroke_color=color, fill_color=PANEL, fill_opacity=0.9)
                cells.add(VGroup(box, Text(value, font=FONT_CODE, font_size=17, color=WHITE).move_to(box)))
            VGroup(*cells[1:]).arrange(RIGHT, buff=0.1)
            cells.arrange(RIGHT, buff=0.18)
            vectors.add(cells)
        vectors.arrange(DOWN, buff=0.25).move_to(UP * 0.38)
        owner = self.pill("one logical thread · four aligned values", AMBER).move_to(DOWN * 1.18)
        tail = Text("未对齐指针回退标量路径；N%4 的尾部由标量代码处理。",
                    font=FONT_CJK, font_size=16, color=MUTED).move_to(DOWN * 1.82)
        traffic = Text("每个元素仍读 A、读 B、写 C：约 12 B/element。",
                       font=FONT_CJK, font_size=16, color=BLUE).move_to(DOWN * 2.30)
        caveat = Text("向量化减少部分逐元素指令开销的机会，但本轮测量未显示所有算子稳定获益。",
                      font=FONT_CJK, font_size=15, color=AMBER).to_edge(DOWN, buff=0.2)
        self.play(FadeIn(title), FadeIn(title_line), run_time=0.55)
        for row in vectors:
            self.play(FadeIn(row, shift=RIGHT * 0.15), run_time=0.35)
        self.play(FadeIn(owner), FadeIn(tail), FadeIn(traffic), FadeIn(caveat), run_time=0.65)
        self.wait(0.9)
        self.play(FadeOut(VGroup(title, title_line, vectors, owner, tail, traffic, caveat)), run_time=0.45)

    def relu_fusion_story(self):
        """Visualize the measured PyTorch graph simplification for ReLU."""
        title = self.header("ReLU 的框架级优化：把多个张量步骤收敛到一个算子")
        old = self.hardware_card("旧表达式", ["比较 x > 0", "检查 NaN", "创建零张量", "嵌套 where 选择"], RED, width=4.0)
        new = self.hardware_card("torch.relu(x)", ["一个高层算子", "一次输出映射", "减少临时张量"], GREEN, width=4.0)
        arrow = Arrow(old.get_right(), new.get_left(), color=AMBER, buff=0.14, stroke_width=4)
        groups = VGroup(old, new).arrange(RIGHT, buff=1.0).move_to(UP * 0.55)
        arrow.put_start_and_end_on(old.get_right(), new.get_left())
        metric = Text("N = 16,777,216 · 记录延迟：约 4.17 ms → 0.92 ms",
                      font=FONT_CODE, font_size=22, color=AMBER).move_to(DOWN * 1.2)
        note = Text("来自本仓库 ReLU PyTorch v1/v2 CSV 与 profiler；两次运行有背景负载差异，作为量级证据。",
                    font=FONT_CJK, font_size=15, color=MUTED).move_to(DOWN * 1.9)
        invariant = Text("消除不必要的中间张量与调度步骤，通常比给同一公式增加线程更有效。",
                         font=FONT_CJK, font_size=17, color=WHITE).move_to(DOWN * 2.55)
        self.play(FadeIn(title), FadeIn(old, shift=LEFT * 0.12), FadeIn(new, shift=RIGHT * 0.12),
                  GrowArrow(arrow), run_time=0.8)
        self.play(Write(metric), FadeIn(note), run_time=0.7)
        self.play(FadeIn(invariant), run_time=0.45)
        self.wait(1.0)
        self.play(FadeOut(VGroup(title, groups, arrow, metric, note, invariant)), run_time=0.5)

    def ending(self):
        formula = Text(self.FORMULA, font=FONT_CODE, font_size=30, color=AMBER)
        formula.scale(min(1, 11.9/formula.width)).move_to(UP * 0.8)
        sentence = Text("数学定义决定每个值；索引映射决定并行工作归属。", font=FONT_CJK, font_size=24, color=WHITE)
        sentence.move_to(DOWN * 0.12)
        tag = self.badge("CUDA v1 walkthrough · source-checked · conceptual hardware view", GREEN)
        tag.to_edge(DOWN, buff=0.55)
        self.play(Write(formula), FadeIn(sentence, shift=UP * 0.1), run_time=0.75)
        self.play(FadeIn(tag), run_time=0.35)
        self.wait(1.3)

    def activation_title(self):
        return f"{self.OPERATOR}：看见输入如何变成输出"

    def activation_caption(self):
        return self.GRAPH_NOTE or "曲线解释单个元素的数学映射；各下标之间仍相互独立。"

    @staticmethod
    def header(text):
        title = Text(text, font=FONT_CJK, font_size=28, color=WHITE)
        title.scale(min(1, 12.7 / title.width)).to_edge(UP, buff=0.28)
        return title

    @staticmethod
    def badge(text, color):
        label = Text(text, font=FONT_CJK, font_size=16, color=color)
        label.scale(min(1, 11.8 / label.width))
        rect = RoundedRectangle(width=label.width + 0.5, height=label.height + 0.28,
                                corner_radius=0.12, stroke_color=color, stroke_width=1.5,
                                fill_color=PANEL, fill_opacity=0.92)
        return VGroup(rect, label.move_to(rect))

    @classmethod
    def pill(cls, text, color):
        label = Text(text, font=FONT_CJK if any(ord(c)>127 for c in text) else FONT_CODE,
                     font_size=17, color=color)
        label.scale(min(1, 11.7 / label.width))
        rect = RoundedRectangle(width=label.width + 0.35, height=label.height + 0.24,
                                corner_radius=0.1, stroke_color=color, stroke_width=1.5,
                                fill_color=PANEL, fill_opacity=0.95)
        return VGroup(rect, label.move_to(rect))

    @staticmethod
    def vector(name, values, color):
        cells = VGroup()
        title = Text(name, font=FONT_CODE, font_size=22, color=color)
        title.move_to(UP * 1.55)
        cells.add(title)
        for i, val in enumerate(values):
            box = RoundedRectangle(width=1.12, height=0.76, corner_radius=0.09,
                                   stroke_color=color, stroke_width=1.7, fill_color=PANEL, fill_opacity=0.95)
            index = Text(f"[{i}]", font=FONT_CODE, font_size=13, color=MUTED).next_to(box, UP, buff=0.08)
            value = Text(val, font=FONT_CODE, font_size=22, color=WHITE).move_to(box)
            cells.add(VGroup(box, index, value))
        VGroup(*cells[1:]).arrange(RIGHT, buff=0.13).move_to(DOWN * 0.1)
        title.move_to(UP * 1.35)
        return cells

    @staticmethod
    def sample_strip(inputs, outputs, compact=False):
        groups = VGroup()
        for i, (src, dst) in enumerate(zip(inputs, outputs)):
            color = (BLUE, VIOLET, AMBER, GREEN)[i % 4]
            src_text = Text(src, font=FONT_CODE, font_size=14 if compact else 17, color=color)
            arrow = Text("→", font=FONT_CODE, font_size=16 if compact else 19, color=MUTED)
            dst_text = Text(dst, font=FONT_CODE, font_size=14 if compact else 17, color=GREEN)
            groups.add(VGroup(src_text, arrow, dst_text).arrange(DOWN if compact else RIGHT, buff=0.07))
        groups.arrange(DOWN if compact else RIGHT, buff=0.16 if compact else 0.22)
        return groups

    @classmethod
    def hardware_card(cls, title, lines, color, width=3.45):
        box = RoundedRectangle(width=width, height=2.0, corner_radius=0.14,
                               stroke_color=color, stroke_width=1.6, fill_color=PANEL, fill_opacity=0.95)
        heading = Text(title, font=FONT_CJK, font_size=19, color=color)
        body = VGroup(*[Text(line, font=FONT_CODE if any(c in line for c in "→=0123456789") else FONT_CJK,
                             font_size=13 if len(line)>29 else 14, color=WHITE) for line in lines])
        body.arrange(DOWN, buff=0.09)
        content = VGroup(heading, body).arrange(DOWN, buff=0.18).move_to(box)
        return VGroup(box, content)
