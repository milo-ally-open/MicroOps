#!/usr/bin/env python3
"""Inspect Manim renders and create review frames/contact sheets under build/."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path


def find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "README.md").is_file() and (parent / "operators").is_dir():
            return parent
    raise RuntimeError("Could not locate OpsFromScratch root (expected README.md and operators/).")


def resolve_video(root: Path, argument: str) -> Path | None:
    path = Path(argument)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    try:
        path.relative_to(root)
    except ValueError:
        print("Error: video must be inside this repository.", file=sys.stderr)
        return None
    if not path.is_file():
        print(f"Error: video does not exist: {path}", file=sys.stderr)
        return None
    return path


def require_tool(name: str) -> str | None:
    location = shutil.which(name)
    if not location:
        print(f"Error: {name} is required for this command; install it in the Manim environment.", file=sys.stderr)
    return location


def probe_video(path: Path, ffprobe: str) -> dict | None:
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration,size,bit_rate,format_name:stream=codec_type,codec_name,profile,width,height,pix_fmt,r_frame_rate,avg_frame_rate,nb_frames,sample_rate,channels",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode:
        print(f"Error: ffprobe could not read the video:\n{result.stderr.strip()}", file=sys.stderr)
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        print(f"Error: invalid ffprobe output: {exc}", file=sys.stderr)
        return None


def video_stream(data: dict) -> dict | None:
    return next((stream for stream in data.get("streams", []) if stream.get("codec_type") == "video"), None)


def print_info(path: Path, data: dict) -> None:
    stream = video_stream(data)
    if not stream:
        print("Error: no video stream found.", file=sys.stderr)
        return
    metadata = data.get("format", {})
    fps = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "unknown"
    try:
        fps = f"{float(Fraction(fps)):.3f} fps"
    except (ValueError, ZeroDivisionError):
        pass
    print(f"File: {path}")
    print(f"Container: {metadata.get('format_name', 'unknown')}")
    print(f"Duration: {float(metadata.get('duration', 0)):.3f} s")
    print(f"Video: {stream.get('codec_name', 'unknown')} {stream.get('profile', '')}".rstrip())
    print(f"Frame: {stream.get('width', '?')}x{stream.get('height', '?')} @ {fps}, {stream.get('pix_fmt', 'unknown')}")
    print(f"Frames: {stream.get('nb_frames', 'not recorded')}")
    print(f"File size: {int(metadata.get('size', 0)):,} bytes")
    if metadata.get("bit_rate"):
        print(f"Bitrate: {int(metadata['bit_rate']):,} bit/s")
    audio = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
    print(f"Audio streams: {len(audio)}")


def audit_video(path: Path, data: dict, ffmpeg: str) -> int:
    """Check metadata and decode the complete video stream without writing output."""
    stream = video_stream(data)
    metadata = data.get("format", {})
    try:
        duration = float(metadata["duration"])
        width = int(stream["width"])
        height = int(stream["height"])
        fps = Fraction(stream.get("avg_frame_rate") or stream["r_frame_rate"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        print("Audit failed: video metadata is incomplete or invalid.", file=sys.stderr)
        return 2
    if duration <= 0 or width <= 0 or height <= 0 or fps <= 0:
        print("Audit failed: duration, dimensions, and frame rate must be positive.", file=sys.stderr)
        return 2

    command = [ffmpeg, "-hide_banner", "-v", "error", "-nostdin", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode:
        print("Audit failed: full video decode reported errors.", file=sys.stderr)
        if result.stderr.strip():
            print(result.stderr.strip(), file=sys.stderr)
        return result.returncode

    print("Audit passed: metadata is valid and the full video stream decodes without FFmpeg errors.")
    print(f"Duration: {duration:.3f} s | Frame: {width}x{height} @ {float(fps):.3f} fps")
    print(f"Audio streams: {sum(s.get('codec_type') == 'audio' for s in data.get('streams', []))}")
    print("Visual quality, technical correctness, and editability still require human review of frames and the source scene.")
    return 0


def build_output_path(root: Path, argument: str | None, default_name: str) -> Path | None:
    base = root / "build" / "manim"
    path = Path(argument) if argument else Path("build/manim/review") / default_name
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    try:
        path.relative_to(base.resolve())
    except ValueError:
        print("Error: generated review artifacts must stay under build/manim/.", file=sys.stderr)
        return None
    if path.exists():
        print(f"Error: output already exists; choose a new path to preserve prior review artifacts: {path}", file=sys.stderr)
        return None
    return path


def parse_times(value: str, duration: float) -> list[float] | None:
    try:
        times = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError:
        print("Error: --times must be comma-separated seconds, for example 0,1.5,4.", file=sys.stderr)
        return None
    if not times:
        print("Error: provide at least one timestamp.", file=sys.stderr)
        return None
    if any(not math.isfinite(t) or t < 0 or t >= duration for t in times):
        print(f"Error: timestamps must be finite and in [0, {duration:.3f}) seconds.", file=sys.stderr)
        return None
    return sorted(set(times))


def extract_frames(video: Path, times: list[float], output_dir: Path, ffmpeg: str) -> int:
    output_dir.mkdir(parents=True, exist_ok=False)
    for index, timestamp in enumerate(times, start=1):
        output = output_dir / f"frame-{index:03d}-{timestamp:09.3f}s.png"
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-n",
            str(output),
        ]
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        if result.returncode or not output.is_file():
            print(f"Error extracting frame at {timestamp:.3f}s:\n{result.stderr.strip()}", file=sys.stderr)
            return result.returncode or 1
        print(output)
    return 0


def contact_sheet(
    video: Path,
    duration: float,
    interval: float,
    columns: int,
    width: int,
    output: Path,
    ffmpeg: str,
) -> int:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("Error: Pillow is required for contact sheets; install it in the dedicated Manim environment.", file=sys.stderr)
        return 2

    times = [index * interval for index in range(math.ceil(duration / interval))]
    if len(times) > 120:
        print("Error: contact sheet would exceed 120 frames; increase --interval.", file=sys.stderr)
        return 2

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        print(f"Error: output already exists; choose a new path: {output}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="manim-review-", dir=output.parent) as temporary:
        frame_dir = Path(temporary)
        frame_paths = []
        for index, timestamp in enumerate(times, start=1):
            frame = frame_dir / f"frame-{index:03d}.png"
            command = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-ss",
                f"{timestamp:.3f}",
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-n",
                str(frame),
            ]
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            if result.returncode or not frame.is_file():
                print(f"Error extracting contact-sheet frame at {timestamp:.3f}s:\n{result.stderr.strip()}", file=sys.stderr)
                return result.returncode or 1
            frame_paths.append((timestamp, frame))

        with Image.open(frame_paths[0][1]) as first:
            frame_height = round(width * first.height / first.width)
        label_height = 30
        rows = math.ceil(len(frame_paths) / columns)
        sheet = Image.new("RGB", (columns * width, rows * (frame_height + label_height)), "#101722")
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.load_default()
        for index, (timestamp, frame_path) in enumerate(frame_paths):
            with Image.open(frame_path) as image:
                image = image.convert("RGB").resize((width, frame_height), Image.Resampling.LANCZOS)
            x = (index % columns) * width
            y = (index // columns) * (frame_height + label_height)
            sheet.paste(image, (x, y))
            draw.rectangle((x, y + frame_height, x + width, y + frame_height + label_height), fill="#243244")
            draw.text((x + 8, y + frame_height + 8), f"t = {timestamp:.2f}s", fill="white", font=font)
        sheet.save(output, format="PNG")
    print(output)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect Manim videos and create review frames under build/manim/.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    info_parser = subparsers.add_parser("info", help="show video codec, dimensions, frame rate, duration, and size")
    info_parser.add_argument("video", help="video file inside this repository")

    audit_parser = subparsers.add_parser("audit", help="validate metadata and decode the complete video stream")
    audit_parser.add_argument("video", help="video file inside this repository")

    frames_parser = subparsers.add_parser("frames", help="extract selected timestamps as PNG frames")
    frames_parser.add_argument("video", help="video file inside this repository")
    frames_parser.add_argument("--times", required=True, help="comma-separated seconds, for example 0,1.5,4")
    frames_parser.add_argument("--output-dir", help="new directory under build/manim/ (default: unique review directory)")

    sheet_parser = subparsers.add_parser("contact-sheet", help="make a labeled overview image from evenly spaced frames")
    sheet_parser.add_argument("video", help="video file inside this repository")
    sheet_parser.add_argument("--interval", type=float, default=2.0, help="seconds between sampled frames (default: 2)")
    sheet_parser.add_argument("--columns", type=int, default=4, help="number of thumbnails per row (default: 4)")
    sheet_parser.add_argument("--width", type=int, default=480, help="thumbnail width in pixels (default: 480)")
    sheet_parser.add_argument("--output", help="PNG path under build/manim/ (default: unique review path)")

    args = parser.parse_args()
    root = find_repo_root()
    video = resolve_video(root, args.video)
    if video is None:
        return 2
    ffprobe = require_tool("ffprobe")
    if not ffprobe:
        return 2
    data = probe_video(video, ffprobe)
    if data is None:
        return 2
    stream = video_stream(data)
    if not stream:
        print("Error: no video stream found.", file=sys.stderr)
        return 2

    if args.command == "info":
        print_info(video, data)
        return 0

    ffmpeg = require_tool("ffmpeg")
    if not ffmpeg:
        return 2
    if args.command == "audit":
        return audit_video(video, data, ffmpeg)
    duration = float(data.get("format", {}).get("duration", 0))
    if not math.isfinite(duration) or duration <= 0:
        print("Error: video duration is missing or invalid.", file=sys.stderr)
        return 2

    if args.command == "frames":
        times = parse_times(args.times, duration)
        if times is None:
            return 2
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        output_dir = build_output_path(root, args.output_dir, f"{video.stem}-frames-{stamp}")
        if output_dir is None:
            return 2
        return extract_frames(video, times, output_dir, ffmpeg)

    if not math.isfinite(args.interval) or args.interval <= 0 or args.columns < 1 or args.width < 64:
        print("Error: interval must be > 0, columns >= 1, and width >= 64.", file=sys.stderr)
        return 2
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    output = build_output_path(root, args.output, f"{video.stem}-contact-{stamp}.png")
    if output is None:
        return 2
    return contact_sheet(video, duration, args.interval, args.columns, args.width, output, ffmpeg)


if __name__ == "__main__":
    raise SystemExit(main())
