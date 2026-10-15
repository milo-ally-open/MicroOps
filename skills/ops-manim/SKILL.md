---
name: ops-manim
description: Design, implement, render, and review Manim Community Edition animations that teach OpsFromScratch CUDA and CANN operators. Use for operator animation work, not ordinary kernel implementation.
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

Use **Manim Community Edition** (Python distribution/CLI name `manim`), not ManimGL (`manimgl`/`manimlib`). Always install it into a dedicated environment named `manim`; never install Manim or its dependencies into Conda `base`, system Python, or the operator runtime environment. If Conda is available, create a separate `manim` Conda environment. If Conda is unavailable but Python is installed, use a separate Python `venv`. A venv still requires Python; if no Python interpreter exists, report that blocker instead of modifying the system or installing into another environment. Do not add Manim to the repository's operator runtime requirements just for animation. Check the selected environment's Python version, `python -m manim --version`, and the bundled `check` command before rendering.

Environment setup examples (run from WSL/Linux; use the equivalent activation command for another shell):

```bash
# Preferred: isolated Conda environment. Never use `conda install` in base.
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

If an existing Conda `base` environment contains useful packages, leave it unchanged. Do not use `--prefix` to put the animation environment inside the repository. For venv, keep the environment under the user's home directory so it cannot be confused with project output or operator dependencies.

Dependency tiers:

- **Required:** a supported Python environment and Manim Community Edition installed in that same environment. Manim installs its Python dependencies, including PyAV, Cairo/Pango bindings and rendering libraries as supported by the platform/package manager.
- **Linux/WSL pip builds:** if compatible wheels are unavailable and dependencies build from source, the Manim guide lists a C/C++ compiler, Python development headers, `pkg-config`, Cairo development headers, and Pango development headers. On Debian/Ubuntu these are typically `build-essential python3-dev pkg-config libcairo2-dev libpango1.0-dev`. Prefer the official Conda route if managing native libraries manually is inconvenient.
- **Optional video utilities:** modern Manim CE (0.19+) uses PyAV for its ordinary render path, so a separate `ffmpeg` executable is not required just to render. Install FFmpeg/ffprobe only when using external muxing/transcoding or when the review workflow needs `ffprobe`.
- **Optional formula rendering:** `Tex`/`MathTex` requires a working LaTeX toolchain and `dvisvgm`, with the LaTeX packages used by the scene (commonly `standalone`, `amsmath`, and `amssymb`). Use plain `Text` for prose when formulas do not require TeX. Chinese inside TeX may also require CTeX; Chinese `Text` needs a CJK font available to Pango. Check the font actually resolves before relying on it.
- **Not required:** CUDA Toolkit, CANN toolkit, GPU/NPU drivers or devices, PyTorch/torch_npu, TTS model downloads, or a Manim MCP server.

The official installation guide recommends an isolated environment and documents pip, Conda, and Docker options. Follow its current OS-specific directions instead of copying old setup commands from videos. Do not install dependencies, download models, or change system configuration without the user's request. If something is missing, report the exact blocker and continue with source-only work when useful. When the user authorizes dependency installation, install only into the dedicated `manim` Conda environment or the dedicated `ops-manim` venv described above; do not interpret that authorization as permission to modify Conda `base` or system Python.

## Bundled workflow helper

The skill includes `scripts/manim_workflow.py`:

```bash
python skills/ops-manim/scripts/manim_workflow.py check
python skills/ops-manim/scripts/manim_workflow.py render operators/add/animation/add_v1.py AddV1Walkthrough --candidate add-v1-a --quality l
python skills/ops-manim/scripts/manim_workflow.py render operators/add/animation/add_v1.py AddV1Walkthrough --candidate add-v1-a --quality m
```

`check` reports the active Python/Manim CE and optional FFmpeg, LaTeX, and font utilities. `render` checks Manim, accepts only animation sources under `operators/<op>/animation/`, invokes the active Python environment, defaults to low preview quality, and routes media into `build/manim/<op>/<source-name>/candidates/<candidate-id>/`. If `--candidate` is omitted, it generates a unique UTC timestamp ID. Use one candidate ID for different quality renders of the same scene; use a fresh ID for each source revision. A render at the same quality cannot overwrite an existing candidate output. Use `--quality l|m|h` to select low, medium, or high output. It does not install packages or launch a media player. Run `--help` for current arguments.

The helper is a persistent, reusable workflow tool shipped with the tracked skill; it is not a one-off animation generator. Keep the copy at `.codex/skills/ops-manim/` in sync when changing the skill or helper. `.codex/` is ignored local configuration; the repository-owned source of truth is `skills/ops-manim/`.

## Render and review loop

1. Confirm the exact source files, scene names, learning objective, and any user preferences. Inspect local Manim dependencies with the helper.
2. Implement the simplest scene that explains the selected code path. Prefer explicit, named objects and small helper functions over a monolithic `construct()` method. Keep representative tensor values deterministic. Break a longer lesson into named `Scene` classes when that helps review or selective rerendering.
3. Render each preview/revision as a distinct candidate under `build/manim/<op>/<scene>/candidates/<candidate-id>/`. Keep all generated videos, frames, and review notes in `build/`; do not copy candidates into the tracked source tree while they are awaiting review. Use a fresh candidate ID for each revision so a new render cannot silently replace an earlier one.
4. Render the requested review quality for the same candidate, or create a new candidate when the scene changes. Inspect the actual video or representative frames for clipping, CJK glyphs, equation legibility, timing, visual continuity, hardware-model clarity, and source-to-scene fidelity. Reading the Python file alone is not visual review.
5. Present the candidate paths and a concise description of what each version shows. Wait for the user to explicitly approve one before adding a video to the repository. After approval, copy only the selected MP4 to `operators/<op>/animation/videos/<scene>-<candidate-id>.mp4`; keep other candidates in ignored `build/`. Never treat a render completing as approval to publish it.
6. If the user gives timestamped feedback, inspect the corresponding frame before editing. Render the revision under a new candidate ID, review it, and wait for approval before publishing.

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

Put final, maintainable Manim scene source and intentional small source assets in `operators/<op>/animation/`. Keep all unapproved videos, frames, scratch scripts, notebooks, generated data, Manim `media/`, TeX cache, and logs under `build/manim/<op>/`; `/build/manim/` is explicitly ignored. Only after the user approves a candidate may its MP4 be copied into `operators/<op>/animation/videos/` as a deliberate repository deliverable. Do not put temporary scripts in `tools/` or alongside final animation source, and do not commit them. The reusable `skills/ops-manim/scripts/manim_workflow.py` is an intentional tracked utility, not a temporary generator. Do not overwrite existing scenes or assets; inspect and edit them in place when the request is to revise an existing animation.

When finished, report the source files changed, scene classes created, exact render status/quality, what was visually reviewed, and any missing dependencies or unverified claims. Never call a render successful if Manim failed or if only source/static checks ran.

## Inspiration

This project-specific workflow takes inspiration from [Yusuke710/manim-skill](https://github.com/Yusuke710/manim-skill), especially its render-review-iterate loop. It is adapted to OpsFromScratch's per-operator source layout; its TTS, viewer, and session-storage tools are not required.
