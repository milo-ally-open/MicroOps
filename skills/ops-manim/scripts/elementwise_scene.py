"""Reusable hardware-aware teaching scene for OpsFromScratch CUDA v1 operators."""

from __future__ import annotations

import math

from manim import *
import numpy as np


BG = "#0B1220"
PANEL = "#152236"
PANEL_2 = "#1C2E45"
WHITE = "#F3F7FC"
MUTED = "#A9B8CA"
BLUE = "#55C7F5"
GREEN = "#57D99A"
AMBER = "#FFC857"
RED = "#FF7B72"
FONT_CJK = "SimSun"
FONT_CODE = "DejaVu Sans Mono"


class ElementwiseV1Scene(Scene):
    """Tell one coherent story: function, lane ownership, hardware path, tail."""

    OPERATOR = "Elementwise"
    FORMULA = "Y[i] = f(X[i])"
    INPUT_LABELS = ()
    OUTPUTS = ()
    CODE = "Y[idx] = f(X[idx]);"
    BINARY = False
    GRAPH_KEY = None
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
        self.show_opening()
        if self.GRAPH_KEY:
            self.show_activation_curve()
        else:
            self.show_elementwise_lanes()
        self.show_cuda_mapping()
        self.show_hardware_dataflow()
        self.show_tail_guard()
        self.show_kernel_excerpt()

    def show_opening(self):
        title = self.header("从一个元素看懂 CUDA 并行计算")
        op = Text(self.OPERATOR, font=FONT_CJK, font_size=33, color=BLUE)
        formula = Text(self.FORMULA, font=FONT_CODE, font_size=32, color=WHITE)
        op.move_to(UP * 0.85)
        formula.next_to(op, DOWN, buff=0.42)
        note = Text("CUDA v1 · float32 · 一线程处理一个元素", font=FONT_CJK, font_size=22, color=MUTED)
        note.next_to(formula, DOWN, buff=0.5)
        self.play(FadeIn(title, shift=DOWN * 0.12))
        self.play(Write(op), Write(formula), FadeIn(note))
        self.wait(1.4)
        self.play(FadeOut(VGroup(title, op, formula, note)))

    def show_activation_curve(self):
        title = self.header(f"{self.OPERATOR}：输入如何经过函数变成输出")
        x_min, x_max = -4, 4
        y_min, y_max = self.GRAPH_Y_RANGE
        axes = Axes(
            x_range=[x_min, x_max, 2],
            y_range=[y_min, y_max, (y_max - y_min) / 4],
            x_length=7.25,
            y_length=4.15,
            axis_config={"color": MUTED, "stroke_width": 2, "include_tip": True, "tip_width": 0.12, "tip_height": 0.12},
        ).move_to(LEFT * 2.35 + DOWN * 0.12)
        # Axes.add_coordinates() uses MathTex and would require a separate LaTeX
        # installation. Plain Text ticks keep rendering self-contained.
        ticks = VGroup()
        for value in (-4, -2, 0, 2, 4):
            ticks.add(Text(str(value), font=FONT_CODE, font_size=13, color=MUTED)
                      .move_to(axes.c2p(value, y_min) + DOWN * 0.18))
        for value in np.linspace(y_min, y_max, 5):
            ticks.add(Text(f"{value:g}", font=FONT_CODE, font_size=12, color=MUTED)
                      .move_to(axes.c2p(x_min, value) + LEFT * 0.31))
        grid = VGroup(
            *[Line(axes.c2p(value, y_min), axes.c2p(value, y_max), color="#243348", stroke_width=1)
              for value in (-4, -2, 0, 2, 4)],
            *[Line(axes.c2p(x_min, value), axes.c2p(x_max, value), color="#243348", stroke_width=1)
              for value in np.linspace(y_min, y_max, 5)],
        )
        curve = axes.plot(self.GRAPH_FUNCTIONS[self.GRAPH_KEY], x_range=[x_min, x_max], color=GREEN, stroke_width=4)
        x_label = Text("输入 x", font=FONT_CJK, font_size=19, color=MUTED).next_to(axes.x_axis, DOWN, buff=0.28)
        y_label = Text("输出 y", font=FONT_CJK, font_size=19, color=MUTED).next_to(axes.y_axis, LEFT, buff=0.22).rotate(PI / 2)

        sample_lines = [f"{source}  →  {output}" for source, output in zip(self.INPUT_LABELS, self.OUTPUTS)]
        sample_group = self.panel("曲线上对应的采样值", sample_lines, BLUE, width=3.9, height=3.25)
        sample_group.move_to(RIGHT * 4.25 + DOWN * 0.15)
        note = Text(self.GRAPH_NOTE or "曲线用于解释数学映射；CUDA 线程仍各自处理独立下标。", font=FONT_CJK, font_size=17, color=AMBER)
        note.to_edge(DOWN, buff=0.22)

        points = VGroup()
        projections = VGroup()
        for value in (-2.0, -0.5, 0.5, 2.0):
            result = self.GRAPH_FUNCTIONS[self.GRAPH_KEY](value)
            point = Dot(axes.c2p(value, result), radius=0.075, color=AMBER)
            guide = DashedLine(axes.c2p(value, 0), point.get_center(), color=AMBER, dash_length=0.07, stroke_width=1.4)
            points.add(point)
            projections.add(guide)

        self.play(FadeIn(title))
        self.play(Create(grid), Create(axes), FadeIn(ticks), FadeIn(x_label), FadeIn(y_label))
        self.play(Create(curve, run_time=1.6))
        self.play(LaggedStart(*[Create(guide) for guide in projections], lag_ratio=0.12),
                  LaggedStart(*[GrowFromCenter(point) for point in points], lag_ratio=0.12))
        self.play(FadeIn(sample_group, shift=LEFT * 0.15), FadeIn(note))
        self.wait(2.0)
        self.play(FadeOut(VGroup(title, axes, grid, ticks, curve, x_label, y_label, sample_group, note, points, projections)))

    def show_elementwise_lanes(self):
        title = self.header("向量运算：同一位置的数据交给同一线程")
        formula = Text(self.FORMULA, font=FONT_CODE, font_size=27, color=AMBER).move_to(UP * 2.35)
        cards = VGroup()
        for lane, (inputs, output) in enumerate(zip(self.INPUT_LABELS, self.OUTPUTS)):
            card = RoundedRectangle(width=2.95, height=3.0, corner_radius=0.12,
                                    stroke_color=BLUE, stroke_width=2, fill_color=PANEL, fill_opacity=1)
            worker = Text(f"thread {lane}", font=FONT_CODE, font_size=18, color=BLUE).move_to(card.get_top() + DOWN * 0.38)
            index = Text(f"idx = {lane}", font=FONT_CODE, font_size=19, color=WHITE).next_to(worker, DOWN, buff=0.3)
            in_value = Text(inputs, font=FONT_CODE, font_size=15, color=BLUE).next_to(index, DOWN, buff=0.42)
            in_value.scale(min(1.0, 2.62 / in_value.width))
            arrow = Arrow(in_value.get_bottom(), in_value.get_bottom() + DOWN * 0.36, buff=0.04, color=AMBER)
            out_value = Text(output, font=FONT_CODE, font_size=16, color=GREEN).next_to(arrow, DOWN, buff=0.12)
            out_value.scale(min(1.0, 2.62 / out_value.width))
            cards.add(VGroup(card, worker, index, in_value, arrow, out_value))
        cards.arrange(RIGHT, buff=0.18).move_to(DOWN * 0.05)
        note = Text("这是四个代表性下标；实际 grid 由 host 按 N 和 block 大小计算。", font=FONT_CJK, font_size=18, color=MUTED)
        note.to_edge(DOWN, buff=0.28)
        self.play(FadeIn(title), Write(formula))
        self.play(LaggedStart(*[FadeIn(card, shift=UP * 0.12) for card in cards], lag_ratio=0.12))
        self.play(FadeIn(note))
        self.wait(2.0)
        self.play(FadeOut(VGroup(title, formula, cards, note)))

    def show_cuda_mapping(self):
        title = self.header("Host 启动 grid，GPU 以 block 为调度单位")
        launch = self.pill("N = 258（示例）   ·   256 threads / block   ·   grid = 2 blocks", BLUE)
        launch.move_to(UP * 2.45)
        grid = self.panel("CUDA grid", ["block 0", "idx 0–255", "block 1", "idx 256–511"], BLUE, 2.8, 2.35)
        sched = self.panel("Block scheduler", ["选择可运行的 block", "执行先后不固定"], AMBER, 3.15, 2.35)
        sm_a = self.panel("SM 示例 A", ["一个 block 在一个 SM 上执行"], GREEN, 2.85, 1.35)
        sm_b = self.panel("SM 示例 B", ["其他 block 的落点不保证"], GREEN, 2.85, 1.35)
        sms = VGroup(sm_a, sm_b).arrange(DOWN, buff=0.25).move_to(RIGHT * 4.45 + DOWN * 0.1)
        grid.move_to(LEFT * 4.55 + DOWN * 0.1)
        sched.move_to(LEFT * 0.3 + DOWN * 0.1)
        arrows = VGroup(Arrow(grid.get_right(), sched.get_left(), buff=0.12, color=BLUE),
                        Arrow(sched.get_right(), sms.get_left(), buff=0.12, color=AMBER))
        index_eq = Text("idx = blockIdx.x × 256 + threadIdx.x", font=FONT_CODE, font_size=22, color=WHITE).move_to(DOWN * 2.15)
        note = Text("N、blockDim 和 gridDim 取自当前 CUDA host；SM 示意不代表实际运行轨迹。", font=FONT_CJK, font_size=17, color=MUTED).to_edge(DOWN, buff=0.18)
        self.play(FadeIn(title), FadeIn(launch))
        self.play(FadeIn(grid), FadeIn(sched), FadeIn(sms))
        self.play(GrowArrow(arrows[0]), GrowArrow(arrows[1]))
        self.play(Write(index_eq), FadeIn(note))
        self.wait(2.0)
        self.play(FadeOut(VGroup(title, launch, grid, sched, sms, arrows, index_eq, note)))

    def show_hardware_dataflow(self):
        title = self.header("一个 warp 的 lane 如何完成 load → compute → store")
        lane_labels = VGroup(*[Text(f"lane {lane}  /  idx {lane}", font=FONT_CODE, font_size=14, color=WHITE)
                               for lane in range(4)])
        lane_labels.add(Text("… lane 31", font=FONT_CODE, font_size=14, color=MUTED))
        lane_labels.arrange(RIGHT, buff=0.22).move_to(UP * 1.9)
        lane_boxes = VGroup()
        for index in range(4):
            lane_boxes.add(self.panel(f"lane {index}", [self.INPUT_LABELS[index], self.OUTPUTS[index]], GREEN, 2.7, 1.48))
        lane_boxes.add(self.panel("…", ["其余 lanes"], MUTED, 2.0, 1.48))
        lane_boxes.arrange(RIGHT, buff=0.14).move_to(UP * 0.58)
        stage = VGroup(
            self.panel("GPU global memory", ["连续输入元素", "A/B 或 X"], BLUE, 2.65, 1.2),
            self.panel("SM · warp lanes", ["lane ℓ 负责 idx ℓ", "32 线程一组（概念）"], GREEN, 2.9, 1.2),
            self.panel("GPU global memory", ["连续输出元素", "C 或 Y"], AMBER, 2.65, 1.2),
        ).arrange(RIGHT, buff=0.7).move_to(DOWN * 1.38)
        arrows = VGroup(Arrow(stage[0].get_right(), stage[1].get_left(), buff=0.08, color=BLUE),
                        Arrow(stage[1].get_right(), stage[2].get_left(), buff=0.08, color=AMBER))
        expr = Text(self.CODE, font=FONT_CODE, font_size=19, color=WHITE)
        expr.scale(min(1.0, 11.8 / expr.width)).move_to(DOWN * 2.5)
        note = Text("连续下标便于合并内存访问；未使用 shared memory。寄存器与指令映射以编译产物为准。", font=FONT_CJK, font_size=16, color=MUTED).to_edge(DOWN, buff=0.15)
        self.play(FadeIn(title), FadeIn(lane_labels))
        self.play(LaggedStart(*[FadeIn(box, shift=UP * 0.1) for box in lane_boxes], lag_ratio=0.08))
        self.play(FadeIn(stage[0]), GrowArrow(arrows[0]), FadeIn(stage[1]), GrowArrow(arrows[1]), FadeIn(stage[2]))
        self.play(Write(expr), FadeIn(note))
        self.wait(2.2)
        self.play(FadeOut(VGroup(title, lane_labels, lane_boxes, stage, arrows, expr, note)))

    def show_tail_guard(self):
        title = self.header("尾块：先算全局下标，再用 idx < N 阻止越界访问")
        subtitle = Text("N = 258   ·   block 1 覆盖 idx 256–511", font=FONT_CODE, font_size=21, color=AMBER).move_to(UP * 2.35)
        lanes = VGroup()
        for thread in range(4):
            idx = 256 + thread
            valid = idx < 258
            color = GREEN if valid else MUTED
            cell = RoundedRectangle(width=2.25, height=1.55, corner_radius=0.12,
                                    stroke_color=color, stroke_width=2, fill_color=PANEL_2 if valid else BG, fill_opacity=1)
            text = VGroup(
                Text(f"threadIdx {thread}", font=FONT_CODE, font_size=15, color=BLUE),
                Text(f"idx = {idx}", font=FONT_CODE, font_size=20, color=WHITE),
                Text("valid" if valid else "masked", font=FONT_CJK, font_size=18, color=color),
            ).arrange(DOWN, buff=0.16).move_to(cell)
            lanes.add(VGroup(cell, text))
        lanes.arrange(RIGHT, buff=0.22).move_to(UP * 0.55)
        guard = Text("if (idx < N) {  ...load, compute, store...  }", font=FONT_CODE, font_size=22, color=AMBER)
        guard.scale(min(1.0, 11.7 / guard.width)).move_to(DOWN * 1.25)
        note = Text("idx 256、257 有效；258、259 不得读取输入或写入输出。仅展开尾块前 4 个线程。", font=FONT_CJK, font_size=18, color=MUTED).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(title), Write(subtitle))
        self.play(LaggedStart(*[FadeIn(lane, shift=UP * 0.15) for lane in lanes], lag_ratio=0.12))
        self.play(Write(guard), FadeIn(note))
        self.wait(2.0)
        self.play(FadeOut(VGroup(title, subtitle, lanes, guard, note)))

    def show_kernel_excerpt(self):
        title = self.header("读懂这几行：host 配置并行度，kernel 负责一个元素")
        host = self.panel("host", ["threads = 256", "blocks = ceil(N / 256)", "kernel<<<blocks, threads>>>(...)"], BLUE, 5.0, 2.3)
        kernel = self.panel("kernel", ["idx = block * 256 + thread", "if idx < N:", self.CODE], GREEN, 6.2, 2.3)
        pair = VGroup(host, kernel).arrange(RIGHT, buff=0.52).move_to(UP * 0.35)
        summary = Text("一个逻辑线程拥有一个输出下标；调度顺序由 GPU 决定。", font=FONT_CJK, font_size=24, color=AMBER).move_to(DOWN * 1.55)
        self.play(FadeIn(title), FadeIn(host), FadeIn(kernel))
        self.play(Write(summary))
        self.wait(2.5)

    @staticmethod
    def header(text):
        return Text(text, font=FONT_CJK, font_size=29, color=WHITE).to_edge(UP, buff=0.32)

    @staticmethod
    def panel(title, lines, color, width, height):
        box = RoundedRectangle(width=width, height=height, corner_radius=0.14,
                               stroke_color=color, stroke_width=2, fill_color=PANEL, fill_opacity=1)
        title_text = Text(title, font=FONT_CJK, font_size=19, color=color)
        title_text.scale(min(1.0, (width - 0.3) / title_text.width))
        body = []
        for i, line in enumerate(lines):
            label = Text(line, font=FONT_CODE if any(ch in line for ch in "[]=<>*/)×") else FONT_CJK,
                         font_size=14 if len(line) > 27 else 16, color=WHITE)
            label.scale(min(1.0, (width - 0.3) / label.width))
            body.append(label)
        content = VGroup(title_text, *body)
        content.arrange(DOWN, buff=0.14).move_to(box)
        return VGroup(box, content)

    @staticmethod
    def pill(text, color):
        label = Text(text, font=FONT_CODE, font_size=18, color=color)
        label.scale(min(1.0, 11.7 / label.width))
        rect = RoundedRectangle(width=label.width + 0.34, height=label.height + 0.2,
                                corner_radius=0.1, stroke_color=color, fill_color=PANEL, fill_opacity=1)
        return VGroup(rect, label.move_to(rect))
