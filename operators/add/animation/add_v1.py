"""Hardware-aware teaching animation for CUDA add_kernel_v1."""

from manim import *


FONT_CJK = "Droid Sans Fallback"
FONT_CODE = "DejaVu Sans Mono"
BG = "#101722"
PANEL = "#172333"
WHITE = "#F2F6FC"
MUTED = "#9EADC2"
BLUE = "#5CC8FF"
GREEN = "#55D6A0"
AMBER = "#FFBE55"
RED = "#FF7B72"


class AddV1HardwareWalkthrough(Scene):
    """Show vector ownership, CUDA scheduling concepts, coalescing, and tail guards."""

    def construct(self):
        self.camera.background_color = BG
        # Source truth: add_host_v1 launches 256 threads/block; the kernel maps
        # one thread to one float and guards idx < N. N=258 is illustrative.
        # Hardware facts follow NVIDIA's CUDA Programming Guide. Scheduler
        # placement is shown as a conceptual relationship, never a run trace.
        self.show_vector_problem()
        self.show_launch_geometry()
        self.show_gpu_hierarchy()
        self.show_warp_dataflow()
        self.show_tail_guard()
        self.show_summary()

    def show_vector_problem(self):
        title = self._title("CUDA 向量加法：每个线程处理一个元素")
        formula = Text(
            "C[i] = A[i] + B[i]",
            font=FONT_CODE,
            font_size=34,
            color=WHITE,
        ).move_to([0, 1.55, 0])
        rows = VGroup(
            self._vector_row("A", ["1", "4", "2", "10", "6", "3", "8", "5"], 3, BLUE),
            self._vector_row("B", ["8", "2", "9", "20", "1", "7", "4", "6"], 3, BLUE),
            self._vector_row("C", ["?", "?", "?", "30", "?", "?", "?", "?"], 3, GREEN),
        ).arrange(DOWN, buff=0.33).move_to([0, -0.35, 0])
        note = Text(
            "演示数据：下标 3 的线程读取 A[3]=10、B[3]=20，并写出 C[3]=30",
            font=FONT_CJK,
            font_size=19,
            color=MUTED,
        ).move_to([0, -2.35, 0])
        self.play(FadeIn(title, shift=DOWN * 0.1), Write(formula))
        self.play(LaggedStart(*[FadeIn(row, shift=UP * 0.12) for row in rows], lag_ratio=0.18))
        self.play(Write(note))
        self.wait(1.1)
        self.play(FadeOut(VGroup(title, formula, rows, note)))

    def show_launch_geometry(self):
        title = self._title("Host 把一维向量切成 thread blocks")
        launch = self._pill("N = 258    blockDim.x = 256    gridDim.x = 2", BLUE)
        launch.move_to([0, 2.05, 0])

        block0 = self._block_card(
            "block 0",
            "threads 0 ... 255",
            "global idx 0 ... 255",
            "覆盖 A/B/C[0 ... 255]",
            BLUE,
        ).move_to([-3.25, 0.25, 0])
        block1 = self._block_card(
            "block 1",
            "thread 0 → idx 256 ✓",
            "thread 1 → idx 257 ✓",
            "thread 2 ... 255 → idx ≥ 258",
            AMBER,
        ).move_to([3.25, 0.25, 0])
        mapping = Text(
            "idx = blockIdx.x * blockDim.x + threadIdx.x",
            font=FONT_CODE,
            font_size=24,
            color=WHITE,
        ).move_to([0, -1.65, 0])
        note = Text(
            "N=258 仅用于讲解边界；host 代码按 ceil(N / 256) 计算 block 数",
            font=FONT_CJK,
            font_size=18,
            color=MUTED,
        ).move_to([0, -2.55, 0])
        self.play(FadeIn(title), FadeIn(launch, shift=DOWN * 0.12))
        self.play(FadeIn(block0, shift=LEFT * 0.18), FadeIn(block1, shift=RIGHT * 0.18))
        self.play(Write(mapping), FadeIn(note))
        self.wait(1.2)
        self.play(FadeOut(VGroup(title, launch, block0, block1, mapping, note)))

    def show_gpu_hierarchy(self):
        title = self._title("从 grid 到 GPU：逻辑分组如何对应硬件")
        grid = self._panel("CUDA grid", ["block 0", "block 1"], BLUE, width=2.8, height=2.0)
        scheduler = self._panel(
            "block scheduler",
            ["分派可运行的 block", "顺序与并发不固定"],
            AMBER,
            width=3.25,
            height=2.0,
        )
        sms = VGroup(
            self._sm_card("SM A"),
            self._sm_card("SM B"),
            Text("…", font=FONT_CODE, font_size=28, color=MUTED),
        ).arrange(DOWN, buff=0.16)
        sms.move_to([4.4, 0.0, 0])
        grid.move_to([-4.55, 0.1, 0])
        scheduler.move_to([-0.3, 0.1, 0])
        arrow_a = Arrow(grid.get_right(), scheduler.get_left(), buff=0.12, color=BLUE)
        arrow_b = Arrow(scheduler.get_right(), sms.get_left(), buff=0.15, color=AMBER)
        detail = self._pill("一个 thread block 的线程在同一个 SM 上执行", GREEN)
        detail.move_to([0, -2.35, 0])
        caveat = Text(
            "图示表达层级关系；不表示本次运行的 SM 落点、执行顺序或同时驻留数量",
            font=FONT_CJK,
            font_size=17,
            color=MUTED,
        ).move_to([0, -3.0, 0])
        self.play(FadeIn(title))
        self.play(FadeIn(grid), FadeIn(scheduler), FadeIn(sms))
        self.play(GrowArrow(arrow_a), GrowArrow(arrow_b))
        self.play(FadeIn(detail), FadeIn(caveat))
        self.wait(1.4)
        self.play(FadeOut(VGroup(title, grid, scheduler, sms, arrow_a, arrow_b, detail, caveat)))

    def show_warp_dataflow(self):
        title = self._title("在 SM 内：一个 block 被组织成 warps")
        geometry = Text(
            "256 threads / block = 8 warps × 32 lanes",
            font=FONT_CODE,
            font_size=26,
            color=BLUE,
        ).move_to([0, 2.15, 0])
        warp_group = VGroup()
        for warp in range(8):
            color = GREEN if warp == 0 else "#35516A"
            bar = RoundedRectangle(
                width=7.3,
                height=0.27,
                corner_radius=0.06,
                stroke_color=color,
                fill_color=color,
                fill_opacity=0.45 if warp == 0 else 0.2,
            )
            label = Text(f"warp {warp}", font=FONT_CODE, font_size=14, color=WHITE if warp == 0 else MUTED)
            label.next_to(bar, LEFT, buff=0.18)
            warp_group.add(VGroup(bar, label))
        warp_group.arrange(DOWN, buff=0.1).move_to([0.25, 0.55, 0])
        lane_strip = self._lane_strip().move_to([0, -1.35, 0])
        caption = Text(
            "展开 warp 0：lane ℓ 对应 threadIdx.x=ℓ（此处是 block 0）",
            font=FONT_CJK,
            font_size=19,
            color=WHITE,
        ).move_to([0, -2.4, 0])
        note = Text(
            "lane 是逻辑线程分组示意；warp 的实际发射/执行时序由硬件调度",
            font=FONT_CJK,
            font_size=16,
            color=MUTED,
        ).move_to([0, -3.05, 0])
        self.play(FadeIn(title), Write(geometry))
        self.play(LaggedStart(*[FadeIn(row) for row in warp_group], lag_ratio=0.08))
        self.play(FadeIn(lane_strip, shift=UP * 0.12), FadeIn(caption), FadeIn(note))
        self.wait(1.4)
        self.play(FadeOut(VGroup(title, geometry, warp_group, lane_strip, caption, note)))

    def show_tail_guard(self):
        title = self._title("连续 lane 访问连续元素；尾部由 guard 保护")
        subtitle = Text(
            "一个 warp 读取 A[i]、B[i]，逐 lane 相加，再写回 C[i]",
            font=FONT_CJK,
            font_size=21,
            color=WHITE,
        ).move_to([0, 2.35, 0])

        lanes = self._lane_strip(active_count=8, show_ellipsis=True).move_to([0, 1.4, 0])
        a_row = self._memory_row("A", BLUE).move_to([0, 0.65, 0])
        b_row = self._memory_row("B", BLUE).move_to([0, -0.05, 0])
        c_row = self._memory_row("C", GREEN).move_to([0, -0.75, 0])
        arrows = VGroup(
            Arrow([-0.1, 0.5, 0], [-0.1, 0.12, 0], buff=0.04, color=BLUE),
            Arrow([-0.1, -0.2, 0], [-0.1, -0.58, 0], buff=0.04, color=GREEN),
        )
        compute = self._pill("lane ℓ:  C[idx] = A[idx] + B[idx]", AMBER).move_to([0, -1.55, 0])
        coalescing = Text(
            "连续下标 → warp 的 global-memory 请求通常可合并（事务数受对齐与架构影响）",
            font=FONT_CJK,
            font_size=17,
            color=BLUE,
        ).move_to([0, -2.25, 0])
        memory = Text(
            "本 kernel 直接读写 global memory；没有显式使用 shared memory",
            font=FONT_CJK,
            font_size=17,
            color=MUTED,
        ).move_to([0, -2.85, 0])
        self.play(FadeIn(title), FadeIn(subtitle))
        self.play(FadeIn(lanes), FadeIn(a_row), FadeIn(b_row), FadeIn(c_row))
        self.play(GrowArrow(arrows[0]), GrowArrow(arrows[1]), FadeIn(compute))
        self.play(Write(coalescing), FadeIn(memory))
        self.wait(1.5)
        self.play(FadeOut(VGroup(title, subtitle, lanes, a_row, b_row, c_row, arrows, compute, coalescing, memory)))

        tail_title = self._title("N=258 的最后一个 block：只让有效元素写回")
        tail_lanes = self._tail_lanes().move_to([0, 0.45, 0])
        code = Text(
            "if (idx < N)  C[idx] = A[idx] + B[idx];",
            font=FONT_CODE,
            font_size=23,
            color=WHITE,
        ).move_to([0, -1.35, 0])
        explanation = Text(
            "block 1 的 lane 0、1 对应 idx 256、257；lane 2 起越界，不读写数组",
            font=FONT_CJK,
            font_size=18,
            color=AMBER,
        ).move_to([0, -2.1, 0])
        self.play(FadeIn(tail_title), FadeIn(tail_lanes))
        self.play(Write(code), FadeIn(explanation))
        self.wait(1.6)
        self.play(FadeOut(VGroup(tail_title, tail_lanes, code, explanation)))

    def show_summary(self):
        title = self._title("把代码、数据和硬件层级合起来看")
        chain = VGroup(
            self._summary_card("Host", "grid = ceil(N / 256)", BLUE),
            self._summary_card("Thread", "idx = block × 256 + lane", AMBER),
            self._summary_card("Element", "读 A/B → 加法 → 写 C", GREEN),
        ).arrange(RIGHT, buff=0.22).move_to([0, 0.9, 0])
        arrows = VGroup(
            Arrow(chain[0].get_right(), chain[1].get_left(), buff=0.04, color=MUTED),
            Arrow(chain[1].get_right(), chain[2].get_left(), buff=0.04, color=MUTED),
        )
        invariant = Text(
            "向量元素彼此独立；尾部线程由 idx < N 排除",
            font=FONT_CJK,
            font_size=26,
            color=GREEN,
        ).move_to([0, -1.0, 0])
        caveat = Text(
            "warp/SM 图示依据 CUDA 编程模型；不是性能测量，也不是某次运行的时间线",
            font=FONT_CJK,
            font_size=17,
            color=MUTED,
        ).move_to([0, -2.1, 0])
        self.play(FadeIn(title), FadeIn(chain), GrowArrow(arrows[0]), GrowArrow(arrows[1]))
        self.play(Write(invariant), FadeIn(caveat))
        self.wait(2)

    @staticmethod
    def _title(text):
        return Text(text, font=FONT_CJK, font_size=30, color=WHITE).to_edge(UP, buff=0.34)

    @staticmethod
    def _pill(text, color):
        label = Text(text, font=FONT_CJK, font_size=19, color=color)
        box = RoundedRectangle(
            width=label.width + 0.5,
            height=label.height + 0.25,
            corner_radius=0.12,
            stroke_color=color,
            fill_color=PANEL,
            fill_opacity=0.9,
        )
        return VGroup(box, label)

    @staticmethod
    def _vector_row(name, values, highlight, color):
        label = Text(name, font=FONT_CODE, font_size=21, color=color).scale_to_fit_width(0.5)
        cells = VGroup()
        for index, value in enumerate(values):
            border = GREEN if index == highlight and name == "C" else color
            box = RoundedRectangle(
                width=1.08,
                height=0.75,
                corner_radius=0.08,
                stroke_color=border,
                fill_color=PANEL,
                fill_opacity=1,
                stroke_width=2 if index == highlight else 1,
            )
            item = VGroup(
                Text(str(value), font=FONT_CODE, font_size=17, color=WHITE),
                Text(str(index), font=FONT_CODE, font_size=10, color=MUTED),
            ).arrange(DOWN, buff=0.01).move_to(box)
            cells.add(VGroup(box, item))
        cells.arrange(RIGHT, buff=0.08)
        return VGroup(label, cells).arrange(RIGHT, buff=0.25)

    @staticmethod
    def _block_card(title, line1, line2, line3, color):
        panel = RoundedRectangle(
            width=5.8,
            height=2.45,
            corner_radius=0.15,
            stroke_color=color,
            stroke_width=2,
            fill_color=PANEL,
            fill_opacity=1,
        )
        content = VGroup(
            Text(title, font=FONT_CODE, font_size=23, color=color),
            Text(line1, font=FONT_CODE, font_size=17, color=WHITE),
            Text(line2, font=FONT_CODE, font_size=17, color=WHITE),
            Text(line3, font=FONT_CJK, font_size=17, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        content.move_to(panel.get_center())
        return VGroup(panel, content)

    @staticmethod
    def _panel(title, lines, color, width, height):
        box = RoundedRectangle(
            width=width,
            height=height,
            corner_radius=0.15,
            stroke_color=color,
            stroke_width=2,
            fill_color=PANEL,
            fill_opacity=1,
        )
        text = VGroup(
            Text(title, font=FONT_CJK, font_size=21, color=color),
            *[Text(line, font=FONT_CJK, font_size=16, color=WHITE) for line in lines],
        ).arrange(DOWN, buff=0.2).move_to(box)
        return VGroup(box, text)

    @staticmethod
    def _sm_card(name):
        return RoundedRectangle(width=2.2, height=0.62, corner_radius=0.1, stroke_color=GREEN, fill_color=PANEL, fill_opacity=1).add(
            Text(name, font=FONT_CODE, font_size=18, color=GREEN)
        )

    @staticmethod
    def _lane_strip(active_count=8, show_ellipsis=False):
        cells = VGroup()
        visible = active_count
        for lane in range(visible):
            box = RoundedRectangle(
                width=0.62,
                height=0.55,
                corner_radius=0.05,
                stroke_color=GREEN if lane < 2 else BLUE,
                fill_color="#183344" if lane < 2 else PANEL,
                fill_opacity=1,
            )
            label = Text(str(lane), font=FONT_CODE, font_size=13, color=WHITE).move_to(box)
            cells.add(VGroup(box, label))
        if show_ellipsis:
            cells.add(Text("…31", font=FONT_CODE, font_size=15, color=MUTED))
        cells.arrange(RIGHT, buff=0.08)
        lane_label = Text("lane", font=FONT_CODE, font_size=15, color=MUTED)
        return VGroup(lane_label, cells).arrange(RIGHT, buff=0.18)

    @staticmethod
    def _memory_row(name, color):
        label = Text(f"{name}[i]", font=FONT_CODE, font_size=17, color=color).scale_to_fit_width(1.0)
        cells = VGroup()
        for index in range(8):
            box = RoundedRectangle(
                width=0.82,
                height=0.42,
                corner_radius=0.05,
                stroke_color=color,
                fill_color=PANEL,
                fill_opacity=1,
            )
            cells.add(VGroup(box, Text(f"{name}[{index}]", font=FONT_CODE, font_size=9, color=WHITE).move_to(box)))
        cells.add(Text("… [31]", font=FONT_CODE, font_size=13, color=MUTED))
        cells.arrange(RIGHT, buff=0.06)
        return VGroup(label, cells).arrange(RIGHT, buff=0.18)

    @staticmethod
    def _tail_lanes():
        lane_cells = VGroup()
        for lane in range(32):
            active = lane < 2
            color = GREEN if active else "#526171"
            box = RoundedRectangle(
                width=0.34,
                height=0.62,
                corner_radius=0.035,
                stroke_color=color,
                fill_color="#183344" if active else "#202A35",
                fill_opacity=1,
                stroke_width=2 if active else 1,
            )
            content = VGroup(
                Text(str(lane), font=FONT_CODE, font_size=9, color=WHITE),
                Text("✓" if active else "×", font=FONT_CJK, font_size=14, color=color),
            ).arrange(DOWN, buff=0.02).move_to(box)
            lane_cells.add(VGroup(box, content))
        lane_cells.arrange(RIGHT, buff=0.035)
        heading = Text("block 1 · warp 0 · lane", font=FONT_CODE, font_size=15, color=MUTED)
        return VGroup(heading, lane_cells).arrange(DOWN, buff=0.2)

    @staticmethod
    def _summary_card(title, line, color):
        box = RoundedRectangle(
            width=3.8,
            height=1.45,
            corner_radius=0.12,
            stroke_color=color,
            fill_color=PANEL,
            fill_opacity=1,
        )
        text = VGroup(
            Text(title, font=FONT_CJK, font_size=22, color=color),
            Text(line, font=FONT_CODE, font_size=15, color=WHITE),
        ).arrange(DOWN, buff=0.22).move_to(box)
        return VGroup(box, text)
