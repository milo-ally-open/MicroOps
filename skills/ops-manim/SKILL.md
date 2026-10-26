---
name: ops-manim
description: Design, implement, render, and review Manim Community Edition animations that teach MicroInfer CUDA and CANN operators. Use for operator animation work, not ordinary kernel implementation.
---

# Ops Manim

Make animations that teach how an operator maps data and work onto its backend. Derive every technical claim from the selected operator's code and docs. The skill is self-contained; use the bundled workflow helper when it saves repeated setup.

## Core mental models

Use these as lenses, not as a requirement to show every concept in every video:

1. **Dataflow before decoration.** Trace input → indexing/work assignment → guards → computation → intermediate values → output. Animate the data dependency that explains the code. Add visual polish only when it reinforces that path.
2. **Logical work is not a hardware timeline.** A CUDA thread index or CANN work item describes logical ownership. It does not establish the order or simultaneity in which hardware executes it. Animate several workers together only to explain independence; label the view as conceptual when needed. Show synchronization or ordering only when the code guarantees it.
3. **Ownership and memory movement.** Make clear which worker owns each element/tile, where values are read from and written to, and when intermediates are reused or discarded. Distinguish host/device copies from kernel reads and writes. Do not make every value travel through shared memory unless the implementation does so.
4. **Shape and index invariants.** Keep tensor axes, layout, shape, and index formulas visible when they explain correctness. Use small illustrative tensors while preserving the real mapping. Clearly label toy values and conceptual scale. For tails, show valid elements separately from masked or inactive work.
5. **Parallel decomposition matches the algorithm.** Elementwise work maps independent indices; reductions create partial results and combine them; convolution reuses input neighborhoods and may tile overlapping regions; attention moves through Q/K/V, score dimensions, masking, softmax axis, and weighted values. Choose only the model that matches the selected kernel; never impose a textbook algorithm over different code.
6. **Floating-point order matters.** A reduction tree may change rounding because floating-point addition is not associative. Show the implementation's actual reduction order when known; otherwise label the combine tree as illustrative and avoid claiming bitwise identity.
7. **An animation explains; tests establish correctness.** Use tests and specs to validate edge cases and semantics. Use benchmark records for measured performance only. A smooth animation, build, or screenshot is not evidence of correctness, memory use, or speed.
8. **Connect the algorithm to plausible hardware.** Make parallel execution visible as lanes, warps, blocks, tiles, cores, and data movement when those concepts explain the kernel. Derive the mapping from the implementation and target-device documentation. Keep logical ownership separate from physical scheduling, and label architecture-dependent diagrams as conceptual.

## Hardware-aware visual explanation

Use a small, concrete tensor or matrix to show what workers do in parallel, then connect that data view to the backend's execution and memory model. The diagram should teach the mapping that the code actually implements, not decorate the scene with generic chip imagery.

- **Vectors and elementwise work:** draw indexed input/output vectors and show a group of logical workers handling distinct indices. For CUDA, connect grid → block → thread and, when useful, group threads into 32-lane warps and blocks onto SMs. Show contiguous lanes accessing contiguous elements when the addresses support coalescing. For Ascend C, show the code's block/tile and vector-element mapping; identify AI Core or vector units only when supported by the selected device documentation and kernel implementation.
- **Matrices and tiled work:** label matrix axes and shapes (for example, A[M,K], B[K,N], C[M,N]); animate the actual tile, index mapping, loop over K, accumulation, and any reuse of loaded rows/columns. Show multiple output elements progressing independently where the implementation permits it. Depict Tensor Core/MMA or Ascend Cube/AIC work only when the selected kernel actually uses those instructions or APIs. A matrix picture must distinguish logical matrix multiplication from the physical fragment/layout used by hardware.
- **Memory and compute path:** trace only real transfers and storage levels. A CUDA kernel may read/write global memory and use registers or shared memory; an Ascend C kernel may move data between GM and UB and invoke vector or cube operations. Include a level only when the source or authoritative target-architecture documentation supports it. Connect arrows to loads, stores, copies, reuse, and synchronization in the code.
- **Truthful scheduling:** a CUDA block may be scheduled on an SM, and an Ascend task may be assigned to an AI Core, but do not imply a permanent one-to-one mapping, simultaneous execution of all blocks, a fixed warp order, or a specific number of resident blocks unless verified. Animate parallel workers as a conceptual snapshot, not a guaranteed hardware timeline. Explain masks, tails, barriers, and synchronization only as implemented.
- **Evidence and uncertainty:** inspect the target kernel/host and operator spec first. For hardware facts not explicit in the code, consult official vendor documentation for the actual architecture. If the chip or mapping is unknown, use a clearly labeled conceptual model and state the assumption; do not present profiling results or architecture-specific behavior as observed facts.

Use these visual tools selectively: color each worker/lane consistently; draw vector indices or matrix tiles; animate load → compute → store; expose reuse with values remaining in a buffer; and show tail lanes as inactive. Small illustrative data should preserve the real indexing rules and be labeled as a toy example when its size differs from the actual launch.

## Storytelling and visual quality

Treat “3Blue1Brown-level” as a request for unusually clear mathematical storytelling, not as a request to copy another creator's exact visual identity, scenes, or branding. Build an original visual language for this repository.

- Give each video one learning question and a narrative arc: pose the question, build an intuition with a concrete visual, connect that intuition to the exact formula, map it to the selected kernel, then leave the viewer with one concise invariant.
- Make each operator's mathematics visually distinct. Examples: addition combines aligned contributions; subtraction reverses a displacement; multiplication changes signed scale; division asks for a quotient; ReLU hinges at zero; GELU and SiLU reveal smooth gates; sigmoid maps a logit to a bounded value; tanh is odd and saturating. These are starting metaphors, not mandatory scenes.
- Prefer transforming a small number of meaningful objects over presenting a succession of unrelated text cards. Let a vector entry become a point on a graph, a graph sample become a thread's value, or an index become a memory access. Keep equations short and introduce their symbols before relying on them.
- Plot the exact function from the selected implementation, with useful axes and a few sample markers that match the displayed values. If the implementation is approximate or uses a numerically stable branch, explain that honestly; do not let visual smoothness imply different code semantics.
- Use motion to explain a relation, decision, mapping, or data dependency. Timing should let each change register; avoid rapid decorative movement, dense paragraphs, generic chip imagery, or repeating the same storyboard with only the operator name changed.
- Preserve visual continuity with a small, consistent palette and typography, readable contrast, deliberate composition, and enough negative space. Use CJK text only after checking that the chosen font renders correctly. Keep code excerpts selective and on screen long enough to read.
- Do a technical and visual review at the requested delivery resolution. Inspect the whole-video contact sheet, extract full-size frames around the opening, transitions, densest labels, hardware mapping, and ending, and watch the render when playback is available. Check clipping, glyphs, contrast, plot scale, value/formula consistency, scene pacing, and claims against source. A successful render or metadata audit alone is not visual approval.

For longer per-operator design notes, see [references/storyboard-quality.md](references/storyboard-quality.md).

### Activation and hardware walkthroughs

- For common activation operators (ReLU, GELU, SiLU, sigmoid, tanh and similar functions), draw the function curve over a labeled input range. Mark representative inputs from the scene and connect each marker to its output value so the graph explains the same mapping as the kernel example. Match the curve to the exact implementation: for example, show the tanh approximation for GELU when that is what the kernel computes, and distinguish sigmoid's stable piecewise evaluation from the single smooth mathematical curve. Call out meaningful regions such as a threshold, zero crossing, saturation, or negative tail when they help explain the function.
- For elementwise CUDA kernels, show how the host chooses launch geometry, how `blockIdx` and `threadIdx` form the global index, and how a logical thread handles its element. Connect a warp's lanes to contiguous input and output positions, then show the kernel's actual load → arithmetic/function → store path. If the input length is not a multiple of the block size, visualize a small tail and the real bounds guard.
- Include the GPU hierarchy when it clarifies the implementation: grid → blocks → scheduler → SM → warp lanes. A block executes on one SM, but never claim a fixed SM assignment, block order, concurrency, register allocation, or memory transaction count without evidence. Label hierarchy and lane drawings as conceptual when they explain the model rather than report a captured run.
- For binary elementwise operators, show both input values at the same index converging on one thread's calculation and the result returning to the matching output index. For unary functions, use the plotted curve and sample table to explain the per-index mapping.
- Adapt the same story to CANN from its actual host/kernel code: show its real block/tile/work-item mapping and supported GM/UB/vector or cube movement. Do not transplant CUDA warps/SMs onto Ascend C scenes.

## Repository discovery and story design

Before editing, check Git status and inspect the target operator's `docs/spec.md`, `docs/usage.md`, `operator.yaml`, selected backend kernel and host files, and relevant tests/examples. Preserve unrelated user changes. If the operator or variant is ambiguous, use repository defaults when unambiguous; otherwise ask one focused question.

Write down the teaching objective in a few short comments in the scene source; a separate plan document is unnecessary unless requested. A useful story usually follows:

1. State the problem and show the input/output tensors.
2. Reveal how logical workers obtain indices or tiles.
3. Animate bounds, masks, synchronization, or layout transformations that affect correctness.
4. Show the core arithmetic and data movement.
5. Assemble the output and summarize the invariant or optimization being taught.

For the current CUDA `add_kernel_v1`, the verified mapping is one logical thread per candidate float: `idx = blockIdx.x * blockDim.x + threadIdx.x`; only `idx < N` proceeds to `C[idx] = A[idx] + B[idx]`. A small non-multiple of block size can demonstrate the final partial block. Do not imply all threads run at once, draw blocks as execution-order guarantees, or claim the picture is to scale. For `v2`, re-read the code and show its vectorized aligned path, scalar fallback, and tail only if explaining those details.

Default presentation choices when the user gives none: Chinese labels, a clean 16:9 frame, restrained high-contrast colors, concise silent teaching animation, and low-quality preview followed by a medium-quality review render. These defaults are adjustable; narration, subtitles, TTS, music, and external assets are opt-in.

## Dependencies and environment

The accelerator toolchains and hardware are not needed to render an explanatory animation. Inspecting the operator source is enough; do not compile or run the CUDA/CANN operator unless the user asks for that as part of validation.

Use **Manim Community Edition** (Python distribution/CLI name `manim`), not ManimGL (`manimgl`/`manimlib`). Respect an environment explicitly selected by the user, including Conda `base`; do not silently create a second environment instead. If the user has not selected one, prefer an isolated Conda environment named `manim`, or a Python `venv` when Conda is unavailable. Do not install packages into any environment without user authorization. Do not add Manim to the repository's operator runtime requirements just for animation. Check the selected environment's Python version, `python -m manim --version`, and the bundled `check` command before rendering.

Environment setup examples (run from WSL/Linux; use the equivalent activation command for another shell):

```bash
# Default only when the user has not selected another environment.
conda create -n manim -c conda-forge python=3.12 manim -y
conda activate manim
python --version
python -m manim --version

# Fallback only when Conda is unavailable: isolated venv outside the repository.
python3 -m venv "$HOME/.venvs/ops-manim"
source "$HOME/.venvs/ops-manim/bin/activate"
python -m pip install --upgrade pip
python -m pip install manim
python -m manim --version
```

If the user explicitly selects `base`, use that environment and check whether Manim is already installed before changing it. If it is missing, install there only when the user authorized environment setup. Otherwise leave the environment unchanged and report the blocker. Do not use `--prefix` to put an animation environment inside the repository. For venv, keep the environment under the user's home directory so it cannot be confused with project output or operator dependencies.

Dependency tiers:

- **Required:** a supported Python environment and Manim Community Edition installed in that same environment. Manim installs its Python dependencies, including PyAV, Cairo/Pango bindings and rendering libraries as supported by the platform/package manager.
- **Linux/WSL pip builds:** if compatible wheels are unavailable and dependencies build from source, the Manim guide lists a C/C++ compiler, Python development headers, `pkg-config`, Cairo development headers, and Pango development headers. On Debian/Ubuntu these are typically `build-essential python3-dev pkg-config libcairo2-dev libpango1.0-dev`. Prefer the official Conda route if managing native libraries manually is inconvenient.
- **Optional video utilities:** modern Manim CE (0.19+) uses PyAV for its ordinary render path, so a separate `ffmpeg` executable is not required just to render. Install FFmpeg/ffprobe only when using external muxing/transcoding or when the review workflow needs `ffprobe`.
- **Optional formula rendering:** `Tex`/`MathTex` requires a working LaTeX toolchain and `dvisvgm`, with the LaTeX packages used by the scene (commonly `standalone`, `amsmath`, and `amssymb`). Use plain `Text` for prose when formulas do not require TeX. Chinese inside TeX may also require CTeX; Chinese `Text` needs a CJK font available to Pango. Check the font actually resolves before relying on it.
- **Not required:** CUDA Toolkit, CANN toolkit, GPU/NPU drivers or devices, PyTorch/torch_npu, TTS model downloads, or a Manim MCP server.

Follow current OS-specific installation guidance rather than copying old setup commands from videos. Do not download models or change system configuration without the user's request. If something is missing, report the exact blocker and continue with source-only work when useful. When the user has specified the environment, that choice takes precedence over the default isolation recommendation.

## Bundled workflow helper

The skill includes `scripts/manim_workflow.py`:

```bash
python skills/ops-manim/scripts/manim_workflow.py check
python skills/ops-manim/scripts/manim_workflow.py render operators/add/animation/add_v1.py AddV1Walkthrough --candidate add-v1-a --quality l
python skills/ops-manim/scripts/manim_workflow.py render operators/add/animation/add_v1.py AddV1Walkthrough --candidate add-v1-a --quality m
```

`check` reports the active Python/Manim CE and optional FFmpeg, LaTeX, and font utilities. `render` checks Manim, accepts only animation sources under `operators/<op>/animation/`, invokes the active Python environment, defaults to low preview quality, and routes media into `build/manim/<op>/<source-name>/candidates/<candidate-id>/`. If `--candidate` is omitted, it generates a unique UTC timestamp ID. Use one candidate ID for different quality renders of the same scene; use a fresh ID for each source revision. A render at the same quality cannot overwrite an existing candidate output. Use `--quality l|m|h` to select low, medium, or high output. It does not install packages or launch a media player. Run `--help` for current arguments.

The helper is a persistent, reusable workflow tool shipped with the tracked skill; it is not a one-off animation generator. Keep the copy at `.codex/skills/ops-manim/` in sync when changing the skill or helper. `.codex/` is ignored local configuration; the repository-owned source of truth is `skills/ops-manim/`.

The companion `scripts/manim_video_review.py` supports repeatable media review:

```bash
python skills/ops-manim/scripts/manim_video_review.py info build/manim/add/add_v1/candidates/add-v1-a/videos/add_v1/720p30/AddV1Walkthrough.mp4
python skills/ops-manim/scripts/manim_video_review.py audit build/manim/add/add_v1/candidates/add-v1-a/videos/add_v1/720p30/AddV1Walkthrough.mp4
python skills/ops-manim/scripts/manim_video_review.py contact-sheet build/manim/add/add_v1/candidates/add-v1-a/videos/add_v1/720p30/AddV1Walkthrough.mp4 --interval 2 --columns 4
python skills/ops-manim/scripts/manim_video_review.py frames build/manim/add/add_v1/candidates/add-v1-a/videos/add_v1/720p30/AddV1Walkthrough.mp4 --times 0,3,8,14
```

`info` reads codec, resolution, frame rate, duration, size, and audio-stream count with `ffprobe`. `audit` validates metadata and decodes the full video stream with FFmpeg; it does not judge visual quality or technical claims. `contact-sheet` samples evenly spaced frames and writes one labeled PNG; `frames` extracts exact timestamps to separate PNGs. Generated review artifacts are restricted to `build/manim/`, and existing paths are never overwritten. Both `ffmpeg` and `ffprobe` are required for audit and frame operations; contact sheets also use Pillow from the active Manim environment. Review the contact sheet first, then inspect individual frames or the video around transitions where text, masks, or data movement may be clipped.

## Render and review loop

1. Confirm the exact source files, scene names, learning objective, and any user preferences. Inspect local Manim dependencies with the helper.
2. Implement the simplest scene that explains the selected code path. Prefer explicit, named objects and small helper functions over a monolithic `construct()` method. Keep representative tensor values deterministic. Break a longer lesson into named `Scene` classes when that helps review or selective rerendering.
3. Render each preview/revision as a distinct candidate under `build/manim/<op>/<scene>/candidates/<candidate-id>/`. Keep all generated videos, frames, and review notes in `build/`; do not copy candidates into the tracked source tree while they are awaiting review. Use a fresh candidate ID for each revision so a new render cannot silently replace an earlier one.
4. Render the requested review quality for the same candidate, or create a new candidate when the scene changes. Use `manim_video_review.py info` and `contact-sheet` to inspect the render; extract exact timestamps around any questionable transition with `frames`. For multi-operator work, inspect every operator separately: a shared helper does not prove that each operator's values, curve, labels, or layout are correct. Check clipping, CJK glyphs, equation legibility, timing, visual continuity, hardware-model clarity, and source-to-scene fidelity. Reading the Python file alone is not visual review.
5. Present the candidate paths and a concise description of what each version shows. Wait for the user to explicitly approve one before adding a video to the repository. After approval, copy only the selected MP4 to `operators/<op>/animation/videos/<scene>-<candidate-id>.mp4`; keep other candidates in ignored `build/`. Never treat a render completing as approval to publish it.
6. If the user gives timestamped feedback, inspect the corresponding frame before editing. Render the revision under a new candidate ID, review it, and wait for approval before publishing.

For a detailed per-operator review gate covering source truth, example arithmetic, hardware mapping, and frame-by-frame QA, read [references/storyboard-quality.md](references/storyboard-quality.md).

After explicit approval, copy the selected quality from the candidate directory without overwriting an existing deliverable. For example:

```bash
mkdir -p operators/add/animation/videos
cp -n build/manim/add/add_v1/candidates/add-v1-a/videos/add_v1/720p30/AddV1Walkthrough.mp4 \
  operators/add/animation/videos/AddV1Walkthrough-add-v1-a.mp4
```

Direct Manim CLI is also valid. The helper's equivalent command is typically:

```bash
python -m manim -ql --media_dir build/manim/add/add_v1/candidates/add-v1-a operators/add/animation/add_v1.py AddV1Walkthrough
```

Use installed CLI `--help` if options vary by Manim version. Avoid preview flags that open a GUI in unattended sessions.

## Source hygiene and delivery

Keep the editable Manim scene `.py` beside the deliverable: an MP4 is a flattened render and its text/shapes are not separate editing layers. Future changes belong in the scene source and must be rendered again. Put final scene source and intentional small source assets in `operators/<op>/animation/`. Keep all unapproved videos, frames, scratch scripts, notebooks, generated data, Manim `media/`, TeX cache, and logs under `build/manim/<op>/`; `/build/manim/` is explicitly ignored. Only after the user approves a candidate may its MP4 be copied into `operators/<op>/animation/videos/` as a deliberate repository deliverable. Do not put temporary scripts in `tools/` or alongside final animation source, and do not commit them. The reusable scripts in `skills/ops-manim/scripts/` are intentional tracked utilities, not temporary generators. Do not overwrite existing scenes or assets; inspect and edit them in place when the request is to revise an existing animation.

When finished, report the source files changed, scene classes created, exact render status/quality, what was visually reviewed, and any missing dependencies or unverified claims. Never call a render successful if Manim failed or if only source/static checks ran.

## Inspiration

This project-specific workflow takes inspiration from [Yusuke710/manim-skill](https://github.com/Yusuke710/manim-skill), especially its render-review-iterate loop. It is adapted to MicroInfer's per-operator source layout; its TTS, viewer, and session-storage tools are not required.
