#!/usr/bin/env python3
"""Check Manim dependencies and render an operator animation into build/."""

from __future__ import annotations

import argparse
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def find_repo_root() -> Path:
    """Find the repo root from either the tracked skill or its .codex copy."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "README.md").is_file() and (parent / "operators").is_dir():
            return parent
    raise RuntimeError("Could not locate OpsFromScratch root (expected README.md and operators/).")


def check_environment() -> bool:
    print(f"Python: {platform.python_version()} ({sys.executable})")
    result = subprocess.run(
        [sys.executable, "-m", "manim", "--version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    version_output = result.stdout.strip()
    manim_ok = result.returncode == 0 and "manim community" in version_output.lower()
    if manim_ok:
        print(f"Manim CE: available ({version_output.splitlines()[0]})")
    elif result.returncode == 0:
        print(f"Manim module responded, but edition was not confirmed: {version_output}")
    else:
        print("Manim CE: MISSING or failed to start in this Python environment")
        if version_output:
            print(version_output)
        print("Install Manim Community Edition into this Python environment, then rerun check.")

    for command, purpose in (
        ("ffmpeg", "optional; needed for external muxing/transcoding"),
        ("ffprobe", "optional; needed for media metadata inspection"),
        ("latex", "optional; needed for Tex/MathTex"),
        ("xelatex", "optional; useful for some CJK TeX setups"),
        ("dvisvgm", "optional; required by the usual Manim TeX pipeline"),
        ("fc-match", "optional; checks font fallback on fontconfig systems"),
    ):
        location = shutil.which(command)
        status = f"found: {location}" if location else f"not found ({purpose})"
        print(f"{command}: {status}")

    print("CUDA/CANN SDKs and accelerator hardware are not needed for Manim rendering.")
    return manim_ok


def render_scene(source_arg: str, scene: str, quality: str, candidate: str | None) -> int:
    root = find_repo_root()
    source = Path(source_arg)
    if not source.is_absolute():
        source = root / source
    source = source.resolve()

    try:
        relative = source.relative_to(root)
    except ValueError:
        print(f"Error: source must be inside this repository: {source}", file=sys.stderr)
        return 2

    if (
        len(relative.parts) < 4
        or relative.parts[0] != "operators"
        or relative.parts[2] != "animation"
        or source.suffix != ".py"
    ):
        print(
            "Error: source must be a Python file under operators/<operator>/animation/.",
            file=sys.stderr,
        )
        return 2
    if not source.is_file():
        print(f"Error: animation source does not exist: {relative}", file=sys.stderr)
        return 2

    # Fail early with a dependency message rather than a long Manim traceback.
    if not check_environment():
        return 2

    operator = relative.parts[1]
    if candidate is None:
        candidate = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    elif not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", candidate) or candidate in {".", ".."}:
        print(
            "Error: candidate ID must start with a letter or digit and contain only letters, digits, '.', '_' or '-'.",
            file=sys.stderr,
        )
        return 2

    media_dir = root / "build" / "manim" / operator / source.stem / "candidates" / candidate
    quality_dir = {"l": "480p15", "m": "720p30", "h": "1080p60"}[quality]
    existing_output = media_dir / "videos" / source.stem / quality_dir / f"{scene}.mp4"
    if existing_output.exists():
        print(
            f"Error: candidate {candidate!r} already has a {quality} render. "
            "Use a new candidate ID to preserve the existing video.",
            file=sys.stderr,
        )
        return 2

    media_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "manim",
        f"-q{quality}",
        "--media_dir",
        str(media_dir),
        str(source),
        scene,
    ]
    print(f"Rendering {relative} :: {scene} at quality {quality}.")
    print(f"Candidate: {candidate}")
    print(f"Media directory: {media_dir}")
    result = subprocess.run(command, cwd=root, check=False)
    if result.returncode:
        print(f"Manim exited with status {result.returncode}.", file=sys.stderr)
    else:
        print("Manim exited successfully. Inspect the rendered media before reporting it as reviewed.")
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check Manim CE dependencies or render an OpsFromScratch operator scene."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check", help="report Manim and optional rendering utilities")

    render_parser = subparsers.add_parser("render", help="render a scene into a distinct build/manim candidate directory")
    render_parser.add_argument("source", help="scene .py under operators/<op>/animation/")
    render_parser.add_argument("scene", help="Manim Scene class name")
    render_parser.add_argument(
        "--quality", choices=("l", "m", "h"), default="l", help="l=preview (default), m=medium, h=high"
    )
    render_parser.add_argument(
        "--candidate",
        help="candidate ID to group quality variants (default: unique UTC timestamp); use a new ID for each revision",
    )

    args = parser.parse_args()
    if args.command == "check":
        return 0 if check_environment() else 2
    return render_scene(args.source, args.scene, args.quality, args.candidate)


if __name__ == "__main__":
    raise SystemExit(main())
