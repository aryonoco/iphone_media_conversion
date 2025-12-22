#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "rich>=14.0",
# ]
# ///
#
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2025 Aryan Ameri
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#

"""
AV1 Video Converter Script

Converts Apple video files (.mov, .m4v) to AV1 format using SVT-AV1 encoder.
Preserves resolution, frame rate, color space, and HDR metadata.
Audio streams are copied without re-encoding (unsupported codecs are skipped).

Usage:
    1. Edit the CONFIGURATION section below
    2. Run: ./convert_to_av1.py
    Or: uv run convert_to_av1.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Self

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

# ╔══════════════════════════════════════════════════════════════════╗
# ║                        CONFIGURATION                              ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Edit these variables before running the script                  ║
# ╚══════════════════════════════════════════════════════════════════╝

SOURCE_DIR: str = "/var/home/admin/Videos/src_videos"
DESTINATION_DIR: str = "/var/home/admin/Videos/dst_videos"

# SVT-AV1 Encoding Settings
PRESET: int = 2  # 1 = slowest/best quality, 13 = fastest/lower quality
CRF: int = 30  # 0 = lossless, 63 = worst quality (18-35 typical range)

# Parallel Processing
MAX_WORKERS: int = 2  # Number of videos to encode simultaneously

# Audio Processing
AUDIO_MODE: str = "copy"  # "copy" = passthrough, "opus" = transcode to Opus

# ═══════════════════════════════════════════════════════════════════
#                        END CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

# Constants
SUPPORTED_EXTENSIONS: Final[frozenset[str]] = frozenset({".mov", ".m4v"})

SUPPORTED_AUDIO_CODECS: Final[frozenset[str]] = frozenset(
    {
        "aac",
        "opus",
        "mp3",
        "flac",
        "vorbis",
        "ac3",
        "eac3",
        "pcm_s16le",
        "pcm_s24le",
        "pcm_s32le",
        "pcm_f32le",
        "pcm_s16be",
        "pcm_s24be",
        "pcm_s32be",
        "pcm_f32be",
        "alac",
        "dts",
    }
)

AV1_CODEC_NAMES: Final[frozenset[str]] = frozenset(
    {
        "av1",
        "libaom-av1",
        "libsvtav1",
        "librav1e",
    }
)

# Rich console for output
console = Console()


# ═══════════════════════════════════════════════════════════════════
#                        EXCEPTIONS
# ═══════════════════════════════════════════════════════════════════


class ValidationError(Exception):
    """Raised when environment validation fails."""


class FFprobeError(Exception):
    """Raised when ffprobe fails to analyze a video."""


class FFmpegError(Exception):
    """Raised when ffmpeg encoding fails."""


# ═══════════════════════════════════════════════════════════════════
#                        DATA MODELS
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True, kw_only=True)
class AudioStream:
    """Immutable audio stream metadata."""

    index: int
    codec: str
    channels: int
    sample_rate: int | None = None
    bitrate: int | None = None  # bits per second from ffprobe
    channel_layout: str | None = None  # e.g., "stereo", "5.1(side)"

    @property
    def is_supported(self) -> bool:
        """Check if codec can be stream-copied to MP4."""
        return self.codec in SUPPORTED_AUDIO_CODECS

    @property
    def bitrate_kbps(self) -> int:
        """Bitrate in kbps for Opus encoding.

        Uses original bitrate if available, otherwise calculates
        from channel count (64 kbps per channel, minimum 64 kbps).
        """
        if self.bitrate:
            min_kbps = max(48 * self.channels, 64)
            return max(self.bitrate // 1000, min_kbps)
        return max(64 * self.channels, 64)


@dataclass(frozen=True, slots=True, kw_only=True)
class VideoInfo:
    """Immutable video file metadata from ffprobe."""

    path: Path
    codec: str
    width: int
    height: int
    fps_num: int
    fps_den: int
    pix_fmt: str
    color_primaries: str | None = None
    color_trc: str | None = None
    colorspace: str | None = None
    has_dolby_vision: bool = False
    audio_streams: tuple[AudioStream, ...] = field(default_factory=tuple)

    @property
    def is_av1(self) -> bool:
        """Check if already AV1 encoded."""
        return self.codec in AV1_CODEC_NAMES

    @property
    def fps(self) -> float:
        """Calculate frames per second."""
        if self.fps_den == 0:
            return 0.0
        return self.fps_num / self.fps_den

    @property
    def has_hdr(self) -> bool:
        """Check if video has HDR color space."""
        hdr_indicators = {"bt2020", "smpte2084", "arib-std-b67"}
        return (
            (self.color_primaries in hdr_indicators)
            or (self.color_trc in hdr_indicators)
            or (self.colorspace is not None and "bt2020" in self.colorspace)
        )

    @property
    def supported_audio_streams(self) -> tuple[AudioStream, ...]:
        """Filter to only supported audio streams."""
        return tuple(s for s in self.audio_streams if s.is_supported)

    @property
    def resolution_label(self) -> str:
        """Human-readable resolution label."""
        if self.width >= 3840:
            return "4K"
        elif self.width >= 2560:
            return "1440p"
        elif self.width >= 1920:
            return "1080p"
        elif self.width >= 1280:
            return "720p"
        else:
            return f"{self.width}x{self.height}"

    @classmethod
    def from_ffprobe(cls, path: Path, data: dict) -> Self:
        """Factory method to parse ffprobe JSON output."""
        video_stream: dict | None = None
        audio_streams: list[AudioStream] = []

        for stream in data.get("streams", []):
            codec_type = stream.get("codec_type")

            if codec_type == "video" and video_stream is None:
                video_stream = stream
            elif codec_type == "audio":
                audio_streams.append(
                    AudioStream(
                        index=stream.get("index", 0),
                        codec=stream.get("codec_name", "unknown"),
                        channels=stream.get("channels", 2),
                        sample_rate=int(stream.get("sample_rate", 0)) or None,
                        bitrate=int(stream.get("bit_rate", 0)) or None,
                        channel_layout=stream.get("channel_layout"),
                    )
                )

        if video_stream is None:
            raise FFprobeError(f"No video stream found in {path}")

        # Parse frame rate (e.g., "120/1" or "30000/1001")
        fps_str = video_stream.get("r_frame_rate", "30/1")
        fps_parts = fps_str.split("/")
        fps_num = int(fps_parts[0]) if fps_parts else 30
        fps_den = int(fps_parts[1]) if len(fps_parts) > 1 else 1

        # Detect Dolby Vision from side_data
        has_dv = False
        for side_data in video_stream.get("side_data_list", []):
            if side_data.get("side_data_type") == "DOVI configuration record":
                has_dv = True
                break

        return cls(
            path=path,
            codec=video_stream.get("codec_name", "unknown"),
            width=video_stream.get("width", 0),
            height=video_stream.get("height", 0),
            fps_num=fps_num,
            fps_den=fps_den,
            pix_fmt=video_stream.get("pix_fmt", "yuv420p"),
            color_primaries=video_stream.get("color_primaries"),
            color_trc=video_stream.get("color_transfer"),
            colorspace=video_stream.get("color_space"),
            has_dolby_vision=has_dv,
            audio_streams=tuple(audio_streams),
        )


# ═══════════════════════════════════════════════════════════════════
#                        VALIDATION
# ═══════════════════════════════════════════════════════════════════


def validate_environment() -> None:
    """Validate ffmpeg/ffprobe exist and directories are valid.

    Raises:
        ValidationError: If environment is invalid.
    """
    # Check ffmpeg
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            check=True,
        )
        if "libsvtav1" not in result.stdout:
            console.print(
                "[yellow]Warning:[/] ffmpeg may not have SVT-AV1 support. "
                "Check with: ffmpeg -encoders | grep svtav1"
            )
    except FileNotFoundError:
        raise ValidationError("ffmpeg not found in PATH")
    except subprocess.CalledProcessError:
        raise ValidationError("ffmpeg failed to run")

    # Check ffprobe
    try:
        subprocess.run(
            ["ffprobe", "-version"],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        raise ValidationError("ffprobe not found in PATH")
    except subprocess.CalledProcessError:
        raise ValidationError("ffprobe failed to run")

    # Validate AUDIO_MODE
    if AUDIO_MODE.lower() not in {"copy", "opus"}:
        raise ValidationError(
            f"Invalid AUDIO_MODE '{AUDIO_MODE}'. Must be 'copy' or 'opus'"
        )

    # Check directories
    source = Path(SOURCE_DIR)
    dest = Path(DESTINATION_DIR)

    if not source.exists():
        raise ValidationError(f"Source directory does not exist: {source}")
    if not source.is_dir():
        raise ValidationError(f"Source path is not a directory: {source}")

    # Create destination if needed
    if not dest.exists():
        dest.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]Created destination directory:[/] {dest}")
    elif not dest.is_dir():
        raise ValidationError(f"Destination path is not a directory: {dest}")


# ═══════════════════════════════════════════════════════════════════
#                        VIDEO PROBING
# ═══════════════════════════════════════════════════════════════════


def scan_videos(source_dir: Path) -> tuple[Path, ...]:
    """Scan directory for video files with supported extensions."""
    videos = [
        f
        for f in source_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return tuple(sorted(videos, key=lambda p: p.name.lower()))


def probe_video(path: Path) -> VideoInfo:
    """Extract video metadata using ffprobe.

    Raises:
        FFprobeError: If probing fails.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",  # Only show errors, not warnings about unknown codecs
                "-analyzeduration",
                "100M",  # Analyze longer to avoid warnings
                "-probesize",
                "100M",
                "-print_format",
                "json",
                "-show_streams",
                "-show_format",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(result.stdout)
        return VideoInfo.from_ffprobe(path, data)
    except subprocess.CalledProcessError as e:
        raise FFprobeError(f"ffprobe failed for {path}: {e.stderr}")
    except json.JSONDecodeError as e:
        raise FFprobeError(f"Failed to parse ffprobe output for {path}: {e}")


def should_process(info: VideoInfo, dest_dir: Path) -> tuple[bool, str]:
    """Determine if video should be processed.

    Returns:
        Tuple of (should_process, reason_if_skipped)
    """
    if info.is_av1:
        return False, "Already AV1"

    output_path = dest_dir / f"{info.path.stem}.mp4"
    if output_path.exists():
        return False, "Output exists"

    return True, ""


# ═══════════════════════════════════════════════════════════════════
#                        FFMPEG COMMAND BUILDER
# ═══════════════════════════════════════════════════════════════════


def _build_audio_args(info: VideoInfo) -> list[str]:
    """Build ffmpeg audio arguments based on AUDIO_MODE.

    Returns:
        List of ffmpeg arguments for audio encoding/copying.
    """
    supported_audio = info.supported_audio_streams

    if not supported_audio:
        return ["-an"]

    args: list[str] = []

    if AUDIO_MODE.lower() == "opus":
        # Opus transcoding at original bitrate
        surround_layouts: list[str] = []

        for idx, stream in enumerate(supported_audio):
            args.extend(
                [
                    "-map",
                    f"0:{stream.index}",
                    f"-c:a:{idx}",
                    "libopus",
                    f"-b:a:{idx}",
                    f"{stream.bitrate_kbps}k",
                    f"-vbr:a:{idx}",
                    "on",
                    f"-compression_level:a:{idx}",
                    "10",
                    f"-application:a:{idx}",
                    "audio",
                ]
            )

            # Surround audio needs mapping_family 1 for proper encoding
            if stream.channels > 2:
                args.extend([f"-mapping_family:a:{idx}", "1"])
                # Determine canonical layout for FFmpeg bug workaround
                layout = {6: "5.1", 8: "7.1"}.get(stream.channels)
                if layout:
                    surround_layouts.append(layout)

        # Apply channel layout filter for surround streams (FFmpeg bug #5718)
        if surround_layouts:
            # Use the first surround layout found
            args.extend(["-af", f"aformat=channel_layouts={surround_layouts[0]}"])
    else:
        # COPY mode - passthrough (current behavior)
        for idx, stream in enumerate(supported_audio):
            args.extend(
                [
                    "-map",
                    f"0:{stream.index}",
                    f"-c:a:{idx}",
                    "copy",
                ]
            )

    return args


def build_ffmpeg_command(
    info: VideoInfo,
    output_path: Path,
    *,
    preset: int,
    crf: int,
) -> tuple[str, ...]:
    """Construct ffmpeg command preserving video properties."""
    # Calculate GOP size (10x framerate, max 300)
    gop_size = min(int(info.fps * 10), 300) if info.fps > 0 else 240

    # Tiling based on resolution
    # For high quality (tune=0), use tile-columns=1 max as recommended by SVT-AV1
    # tile-columns=2 is only beneficial for real-time/fast encodes
    if info.width >= 1920:  # 1080p and above
        tile_columns, tile_rows = 1, 0
    else:
        tile_columns, tile_rows = 0, 0

    # Build SVT-AV1 params string (2025 best practices for live action)
    # Reference: https://wiki.x266.mov/blog/svt-av1-fourth-deep-dive-p2
    svtav1_params_list = [
        "tune=0",  # Psychovisual optimization (VQ mode)
        "enable-variance-boost=1",  # Better low-contrast areas (skin, clouds)
        "enable-overlays=1",  # Detail preservation on alt-ref frames
        "enable-tf=1",  # Temporal filtering enabled
        "tf-strength=1",  # Reduced from default 3 to prevent blur/blocking
        "enable-qm=1",  # Enable quantization matrices for efficiency
        "qm-min=0",  # Lower flatness = better compression
        "film-grain=8",  # Preserve natural grain (live action)
        "scm=0",  # Screen content mode OFF for camera footage
        f"tile-columns={tile_columns}",
        f"tile-rows={tile_rows}",
    ]

    # Add luminance-qp-bias ONLY for HLG content (iPhone)
    # Do NOT use for PQ/HDR10 as it breaks quality
    if info.color_trc == "arib-std-b67":  # HLG
        svtav1_params_list.append("luminance-qp-bias=40")

    svtav1_params = ":".join(svtav1_params_list)

    # Building command
    cmd: list[str] = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-stats",
        # Increase analysis duration to avoid warnings about unknown codecs
        "-analyzeduration",
        "100M",
        "-probesize",
        "100M",
        "-i",
        str(info.path),
        "-map",
        "0:v:0",
        "-c:v",
        "libsvtav1",
        "-preset",
        str(preset),
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p10le",  # Always 10-bit for quality
        "-g",
        str(gop_size),
        "-svtav1-params",
        svtav1_params,
        "-dolbyvision",
        "auto",  # Auto-preserve Dolby Vision Profile 8 from iPhone
    ]

    # Add color space preservation
    if info.color_primaries:
        cmd.extend(["-color_primaries", info.color_primaries])
    if info.color_trc:
        cmd.extend(["-color_trc", info.color_trc])
    if info.colorspace:
        cmd.extend(["-colorspace", info.colorspace])

    # Map and encode/copy audio streams
    cmd.extend(_build_audio_args(info))

    # Output settings
    cmd.extend(
        [
            "-map_metadata",
            "0",
            "-movflags",
            "+faststart",
            "-y",  # Overwrite (we already checked for existence)
            str(output_path),
        ]
    )

    return tuple(cmd)


# ═══════════════════════════════════════════════════════════════════
#                        VIDEO CONVERSION
# ═══════════════════════════════════════════════════════════════════


def convert_video(info: VideoInfo, dest_dir: Path, preset: int, crf: int) -> Path:
    """Convert single video to AV1, returning output path.

    Raises:
        FFmpegError: If encoding fails.
    """
    output_path = dest_dir / f"{info.path.stem}.mp4"

    cmd = build_ffmpeg_command(info, output_path, preset=preset, crf=crf)

    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Show progress
        )
        return output_path
    except subprocess.CalledProcessError as e:
        # Clean up partial file
        if output_path.exists():
            output_path.unlink()
        raise FFmpegError(
            f"Encoding failed for {info.path.name}: exit code {e.returncode}"
        )


def convert_all(
    videos: list[tuple[VideoInfo, Path]],
    max_workers: int,
    preset: int,
    crf: int,
) -> None:
    """Convert all videos with parallel execution.

    Raises:
        FFmpegError: On first encoding failure (stops all).
    """
    dest_dir = Path(DESTINATION_DIR)

    console.print(f"\n[bold blue]Starting conversion of {len(videos)} video(s)[/]")
    console.print(f"  Preset: {preset}, CRF: {crf}, Workers: {max_workers}\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all conversion tasks
        future_to_video: dict[Future[Path], VideoInfo] = {
            executor.submit(convert_video, info, dest_dir, preset, crf): info
            for info, _ in videos
        }

        completed = 0
        total = len(videos)

        for future in as_completed(future_to_video):
            video = future_to_video[future]
            completed += 1

            try:
                output_path = future.result()
                console.print(
                    f"[green]✓[/] [{completed}/{total}] "
                    f"{video.path.name} → {output_path.name}"
                )
            except FFmpegError as e:
                console.print(f"[red]✗[/] [{completed}/{total}] {e}")
                # Cancel remaining futures
                for f in future_to_video:
                    f.cancel()
                raise


# ═══════════════════════════════════════════════════════════════════
#                        MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════


def main() -> None:
    """Script entry point."""
    console.print("\n[bold]AV1 Video Converter[/] (SVT-AV1)\n")

    try:
        # Validate environment
        validate_environment()

        source_dir = Path(SOURCE_DIR)
        dest_dir = Path(DESTINATION_DIR)

        # Scan for videos
        video_paths = scan_videos(source_dir)
        if not video_paths:
            console.print(f"[yellow]No .mov or .m4v files found in {source_dir}[/]")
            return

        console.print(f"Found {len(video_paths)} video file(s) to analyze...\n")

        # Probe videos and filter
        videos_to_process: list[tuple[VideoInfo, Path]] = []
        skipped: list[tuple[str, str]] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing videos...", total=len(video_paths))

            for path in video_paths:
                try:
                    info = probe_video(path)
                    should, reason = should_process(info, dest_dir)

                    if should:
                        output_path = dest_dir / f"{info.path.stem}.mp4"
                        videos_to_process.append((info, output_path))
                    else:
                        skipped.append((path.name, reason))

                except FFprobeError as e:
                    console.print(f"[red]Error probing {path.name}:[/] {e}")
                    raise

                progress.advance(task)

        # Display summary table
        if videos_to_process or skipped:
            table = Table(title="Video Analysis Results")
            table.add_column("File", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Resolution")
            table.add_column("FPS")
            table.add_column("Codec")
            table.add_column("HDR")
            table.add_column("DV")  # Dolby Vision
            table.add_column("Audio Streams")

            for info, _ in videos_to_process:
                audio_count = len(info.supported_audio_streams)
                total_audio = len(info.audio_streams)
                audio_str = (
                    f"{audio_count}/{total_audio}"
                    if total_audio > audio_count
                    else str(audio_count)
                )

                table.add_row(
                    info.path.name,
                    "[green]To Convert[/]",
                    info.resolution_label,
                    f"{info.fps:.2f}",
                    info.codec.upper(),
                    "[yellow]Yes[/]" if info.has_hdr else "No",
                    "[magenta]Yes[/]" if info.has_dolby_vision else "No",
                    audio_str,
                )

            for name, reason in skipped:
                table.add_row(
                    name,
                    f"[dim]{reason}[/]",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                )

            console.print(table)

        if not videos_to_process:
            console.print("\n[yellow]No videos to process.[/]")
            return

        # Convert videos
        convert_all(videos_to_process, MAX_WORKERS, PRESET, CRF)

        console.print(
            f"\n[bold green]Done![/] Converted {len(videos_to_process)} video(s)."
        )

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/]")
        sys.exit(130)
    except FFmpegError as e:
        console.print(f"\n[red]Encoding failed:[/] {e}")
        sys.exit(1)
    except FFprobeError as e:
        console.print(f"\n[red]Analysis failed:[/] {e}")
        sys.exit(1)
    except ValidationError as e:
        console.print(f"\n[red]Configuration error:[/] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
