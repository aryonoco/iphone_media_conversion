#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "rich>=14.0",
#     "numpy>=1.26",
#     "scipy>=1.12",
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
Media Converter Script

Converts Apple video files (.mov, .m4v) to AV1 format using SVT-AV1 encoder.
Processes iPhone 16/17 Pro ProRAW DNG files to lossless JPEG XL.
Copies other image formats (HEIC, HEIF, AVIF, JPEG, PNG) as-is with metadata.

Video: Preserves resolution, frame rate, color space, and HDR metadata.
       Audio streams are copied without re-encoding (unsupported codecs skipped).
Images: DNG with JPEG XL compression → processed to lossless .jxl
        (uses dcraw_emu_dng with DNG SDK for full RAW processing)
        Other images and older DNGs → copied unchanged.

Prerequisites:
    - Run libraw_dng.py first to build dcraw_emu_dng
    - Requires: ffmpeg, exiftool, cjxl

Usage:
    1. Edit the CONFIGURATION section below
    2. Run: ./export_media.py
    Or: uv run export_media.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Self

import numpy as np
from scipy.interpolate import PchipInterpolator

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
MAX_WORKERS: int = 6  # Number of videos to encode simultaneously

# Audio Processing
AUDIO_MODE: str = "copy"  # "copy" = passthrough, "opus" = transcode to Opus

# Tone Curve Style for DNG Processing
# Controls the visual "look" of converted ProRAW images
TONE_CURVE_STYLE: str = "apple"  # "apple", "flat", "vivid", "film", "linear"

# ═══════════════════════════════════════════════════════════════════
#                        END CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

# Constants - Video
SUPPORTED_VIDEO_EXTENSIONS: Final[frozenset[str]] = frozenset({".mov", ".m4v"})

# Constants - Image
SUPPORTED_IMAGE_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".jpg", ".jpeg", ".heic", ".heif", ".avif", ".png", ".dng"}
)

# Image formats that should be copied without processing
COPY_ONLY_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".jpg", ".jpeg", ".heic", ".heif", ".avif", ".png"}
)

# Tone curve presets for different looks
# Format: list of (input, output) control points, normalized 0-1
# None = extract from DNG (apple) or skip curve entirely (linear)
TONE_CURVE_PRESETS: Final[dict[str, list[tuple[float, float]] | None]] = {
    "apple": None,  # Extract ProfileToneCurve from DNG metadata
    "flat": [(0.0, 0.0), (1.0, 1.0)],  # Linear passthrough
    "vivid": [
        # Velvia-inspired: steep S-curve, deep blacks, punchy contrast
        (0.00, 0.00),
        (0.10, 0.05),  # Deep shadows
        (0.25, 0.20),  # Shadow compression
        (0.50, 0.50),  # Midpoint anchor
        (0.75, 0.80),  # Highlight lift
        (0.90, 0.95),  # Bright highlights
        (1.00, 1.00),
    ],
    "film": [
        # Portra-inspired: gentle curve, lifted shadows, soft contrast
        (0.00, 0.02),  # Lifted blacks
        (0.10, 0.13),  # Shadow lift
        (0.25, 0.28),  # Gentle toe
        (0.50, 0.52),  # Slight midtone lift
        (0.75, 0.76),  # Gentle shoulder
        (0.90, 0.89),  # Soft highlight rolloff
        (1.00, 0.98),  # Compressed whites
    ],
    "linear": None,  # No curve applied (with -W flag)
}

# Saturation adjustments per style (1.0 = no change)
SATURATION_ADJUSTMENTS: Final[dict[str, float]] = {
    "apple": 1.15,  # +15% to compensate for missing ProfileGainTableMap
    "flat": 1.0,  # Neutral
    "vivid": 1.10,  # +10% (high contrast curve adds natural saturation)
    "film": 0.95,  # -5% (Portra is desaturated)
    "linear": 1.0,
}

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


class ExiftoolError(Exception):
    """Raised when exiftool fails to analyze an image."""


class JxlExtractionError(Exception):
    """Raised when JXL extraction from DNG fails."""


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


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageInfo:
    """Immutable image file metadata from exiftool."""

    path: Path
    format: str  # "dng", "heic", "jpeg", etc.
    width: int
    height: int
    bit_depth: int | None = None
    compression: str | None = None  # "JPEG XL", "JPEG", None
    color_space: str | None = None  # "Display P3", "sRGB"

    @property
    def has_jxl(self) -> bool:
        """Check if DNG contains JPEG XL compression."""
        return self.compression is not None and "jpeg xl" in self.compression.lower()

    @property
    def should_extract_jxl(self) -> bool:
        """Check if we should extract JXL from this DNG."""
        return self.format == "dng" and self.has_jxl

    @property
    def should_copy(self) -> bool:
        """Check if file should be copied without processing."""
        return not self.should_extract_jxl

    @property
    def output_extension(self) -> str:
        """Determine output file extension."""
        if self.should_extract_jxl:
            return ".jxl"
        return self.path.suffix

    @classmethod
    def from_exiftool(cls, path: Path, data: dict) -> Self:
        """Factory method to parse exiftool JSON output."""
        # Determine format from extension
        fmt = path.suffix.lower().lstrip(".")

        # Parse dimensions - handle both string and int
        width = data.get("ImageWidth", 0)
        height = data.get("ImageHeight", 0)
        if isinstance(width, str):
            width = int(width) if width.isdigit() else 0
        if isinstance(height, str):
            height = int(height) if height.isdigit() else 0

        # Parse bit depth
        bits = data.get("BitsPerSample")
        if isinstance(bits, str):
            # Handle "10 10 10" format for RGB
            parts = bits.split()
            bit_depth = int(parts[0]) if parts and parts[0].isdigit() else None
        elif isinstance(bits, int):
            bit_depth = bits
        else:
            bit_depth = None

        return cls(
            path=path,
            format=fmt,
            width=width,
            height=height,
            bit_depth=bit_depth,
            compression=data.get("Compression"),
            color_space=data.get("ColorSpace"),
        )


# ═══════════════════════════════════════════════════════════════════
#                        VALIDATION
# ═══════════════════════════════════════════════════════════════════


def validate_environment() -> None:
    """Validate required tools exist and directories are valid.

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

    # Check exiftool (for image metadata)
    try:
        subprocess.run(
            ["exiftool", "-ver"],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        raise ValidationError("exiftool not found in PATH")
    except subprocess.CalledProcessError:
        raise ValidationError("exiftool failed to run")

    # Check cjxl (for JPEG XL encoding with color space support)
    try:
        subprocess.run(
            ["cjxl", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        raise ValidationError(
            "cjxl not found in PATH. Install libjxl: "
            "https://github.com/libjxl/libjxl"
        )
    except subprocess.CalledProcessError:
        raise ValidationError("cjxl failed to run")

    # Check dcraw_emu_dng (for iPhone ProRAW DNG processing)
    script_dir = Path(__file__).resolve().parent
    dcraw_emu = script_dir / "dcraw_emu_dng"
    if not dcraw_emu.exists():
        raise ValidationError(
            f"dcraw_emu_dng not found at {dcraw_emu}. "
            "Run libraw_dng.py to build it."
        )

    # Test that dcraw_emu_dng can actually run with required libraries
    try:
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = _get_dcraw_library_path()
        result = subprocess.run(
            [str(dcraw_emu)],
            capture_output=True,
            text=True,
            env=env,
            timeout=5,
        )
        # Exit code 1 is expected (no input file), but it means binary runs
        # Check for library loading errors
        if "cannot open shared object file" in result.stderr:
            raise ValidationError(
                f"dcraw_emu_dng missing libraries: {result.stderr.strip()}"
            )
    except subprocess.TimeoutExpired:
        raise ValidationError("dcraw_emu_dng timed out during validation")

    # Validate AUDIO_MODE
    if AUDIO_MODE.lower() not in {"copy", "opus"}:
        raise ValidationError(
            f"Invalid AUDIO_MODE '{AUDIO_MODE}'. Must be 'copy' or 'opus'"
        )

    # Validate TONE_CURVE_STYLE
    valid_styles = {"apple", "flat", "vivid", "film", "linear"}
    if TONE_CURVE_STYLE.lower() not in valid_styles:
        raise ValidationError(
            f"Invalid TONE_CURVE_STYLE '{TONE_CURVE_STYLE}'. "
            f"Must be one of: {', '.join(sorted(valid_styles))}"
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
        if f.is_file() and f.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
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
#                        IMAGE PROBING
# ═══════════════════════════════════════════════════════════════════


def scan_images(source_dir: Path) -> tuple[Path, ...]:
    """Scan directory for image files with supported extensions."""
    images = [
        f
        for f in source_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    return tuple(sorted(images, key=lambda p: p.name.lower()))


def probe_image(path: Path) -> ImageInfo:
    """Extract image metadata using exiftool.

    Raises:
        ExiftoolError: If probing fails.
    """
    try:
        result = subprocess.run(
            [
                "exiftool",
                "-j",
                "-Compression",
                "-ImageWidth",
                "-ImageHeight",
                "-BitsPerSample",
                "-ColorSpace",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(result.stdout)
        if not data:
            raise ExiftoolError(f"No metadata found for {path}")
        return ImageInfo.from_exiftool(path, data[0])
    except subprocess.CalledProcessError as e:
        raise ExiftoolError(f"exiftool failed for {path}: {e.stderr}")
    except json.JSONDecodeError as e:
        raise ExiftoolError(f"Failed to parse exiftool output for {path}: {e}")


def should_process_image(info: ImageInfo, dest_dir: Path) -> tuple[bool, str]:
    """Determine if image should be processed.

    Returns:
        Tuple of (should_process, reason_if_skipped)
    """
    if info.should_extract_jxl:
        output_path = dest_dir / f"{info.path.stem}.jxl"
    else:
        output_path = dest_dir / info.path.name

    if output_path.exists():
        return False, "Output exists"

    return True, ""


# ═══════════════════════════════════════════════════════════════════
#                        RUNTIME HELPERS
# ═══════════════════════════════════════════════════════════════════


def _detect_homebrew_prefix() -> Path | None:
    """Detect Homebrew prefix if available.

    Returns:
        Path to Homebrew prefix, or None if not found.
    """
    try:
        result = subprocess.run(
            ["brew", "--prefix"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return Path(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def _get_dcraw_library_path() -> str:
    """Build LD_LIBRARY_PATH for dcraw_emu_dng runtime libraries.

    The dcraw_emu_dng binary requires several shared libraries at runtime:
    - libraw.so.24: LibRaw library (usually ~/.local/lib)
    - libjpeg.so.8: JPEG library (from Homebrew)
    - libc++.so, libomp.so: LLVM runtime (from Homebrew LLVM)
    - libjxl.so, brotli libs: JPEG XL support (from build deps)

    Returns:
        Colon-separated library path string.
    """
    paths: list[str] = []

    # LibRaw installed location
    libraw_lib = Path.home() / ".local" / "lib"
    if libraw_lib.exists():
        paths.append(str(libraw_lib))

    # libjxl and dependencies from build
    deps_lib = Path.home() / "libraw-dng-build" / "deps" / "lib"
    deps_lib64 = Path.home() / "libraw-dng-build" / "deps" / "lib64"
    if deps_lib.exists():
        paths.append(str(deps_lib))
    if deps_lib64.exists():
        paths.append(str(deps_lib64))

    # Homebrew LLVM (libc++, libomp) and libraries (libjpeg)
    brew_prefix = _detect_homebrew_prefix()
    if brew_prefix:
        llvm_lib = brew_prefix / "opt" / "llvm" / "lib"
        if llvm_lib.exists():
            paths.append(str(llvm_lib))
        brew_lib = brew_prefix / "lib"
        if brew_lib.exists():
            paths.append(str(brew_lib))

    # Preserve existing LD_LIBRARY_PATH
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    if existing:
        paths.append(existing)

    return ":".join(paths)


def _get_baseline_exposure(dng_path: Path) -> float:
    """Read BaselineExposure from DNG and convert to linear multiplier.

    iPhone ProRAW DNG files contain per-image BaselineExposure values
    (typically 0.02 to 2.36 EV) that indicate how much to brighten the image.
    dcraw_emu's -aexpo flag expects a linear multiplier, not EV stops.

    Formula: linear_shift = 2^(baseline_exposure_ev)

    Returns:
        Linear multiplier for exposure correction (1.0 = no change).
    """
    try:
        result = subprocess.run(
            ["exiftool", "-BaselineExposure", "-n", "-s3", str(dng_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        baseline_ev = float(result.stdout.strip())
        # Convert EV stops to linear multiplier
        return 2 ** baseline_ev
    except (subprocess.CalledProcessError, ValueError):
        # Default: no exposure adjustment
        return 1.0


def _extract_tone_curve(dng_path: Path) -> list[tuple[float, float]]:
    """Extract ProfileToneCurve as list of (x, y) control points.

    iPhone ProRAW DNG files contain an embedded tone curve that creates
    the characteristic "Apple look" - shadows compressed, midtones lifted.

    Uses exiv2 to extract the curve data (exiftool truncates large arrays).

    Returns:
        List of (input, output) control points, normalized 0-1.
        Falls back to linear [(0,0), (1,1)] if extraction fails.
    """
    try:
        result = subprocess.run(
            ["exiv2", "-pa", str(dng_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        # Parse "Exif.Image.ProfileToneCurve Float 514 0 0 0.00390625 0.0000118..."
        for line in result.stdout.splitlines():
            if "ProfileToneCurve" in line:
                parts = line.split()
                # Find the start of float values (after "Float" and count)
                float_start = None
                for i, p in enumerate(parts):
                    if p == "Float":
                        float_start = i + 2  # Skip "Float" and count
                        break
                if float_start is None:
                    continue

                values = [float(x) for x in parts[float_start:]]
                if len(values) >= 4:
                    return [(values[i], values[i + 1]) for i in range(0, len(values), 2)]

        return [(0.0, 0.0), (1.0, 1.0)]  # Linear fallback
    except (subprocess.CalledProcessError, ValueError, IndexError):
        return [(0.0, 0.0), (1.0, 1.0)]  # Linear fallback


def _build_tone_curve_lut(
    points: list[tuple[float, float]], bits: int = 16
) -> np.ndarray:
    """Build lookup table using monotonic cubic interpolation.

    Uses PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) for
    smooth monotonic interpolation that won't overshoot the control points.

    Args:
        points: List of (input, output) control points, normalized 0-1.
        bits: Output bit depth (default 16 for 16-bit PPM).

    Returns:
        Lookup table as uint16 array with 2^bits entries.
    """
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])

    # PCHIP = Piecewise Cubic Hermite Interpolating Polynomial (monotonic)
    interp = PchipInterpolator(x, y)

    max_val = (1 << bits) - 1
    lut_x = np.linspace(0, 1, max_val + 1)
    lut_y = np.clip(interp(lut_x), 0, 1)

    return (lut_y * max_val).astype(np.uint16)


def _apply_tone_curve_to_ppm(ppm_path: Path, lut: np.ndarray) -> None:
    """Apply tone curve LUT to PPM while preserving hue.

    Uses luminance-based scaling for hue preservation:
    1. Calculate BT.709 luminance: L = 0.2126*R + 0.7152*G + 0.0722*B
    2. Apply curve to luminance: L' = curve(L)
    3. Scale all channels by ratio: scale = L' / L
    4. Output: R*scale, G*scale, B*scale

    This preserves hue by applying the same multiplier to all channels,
    unlike per-channel application which shifts hues in saturated areas.

    Args:
        ppm_path: Path to 16-bit PPM file (modified in place).
        lut: Lookup table from _build_tone_curve_lut().
    """
    # Read PPM header and data
    with open(ppm_path, "rb") as f:
        # Parse PPM header (handles comments)
        magic = f.readline()
        if magic.strip() != b"P6":
            raise ValueError(f"Not a binary PPM file: {ppm_path}")

        # Skip comments and read dimensions
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        dims = line.decode().split()
        width, height = int(dims[0]), int(dims[1])

        # Read max value
        maxval_line = f.readline()
        maxval = int(maxval_line.decode().strip())

        # Read pixel data
        data = f.read()

    if maxval != 65535:
        raise ValueError(f"Expected 16-bit PPM (maxval 65535), got {maxval}")

    # Parse as big-endian 16-bit unsigned integers
    img = np.frombuffer(data, dtype=">u2").reshape(height, width, 3).astype(np.float32)

    # Calculate BT.709 luminance
    lum = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]

    # Clip luminance to valid LUT range and apply curve
    lum_clipped = np.clip(lum, 0, 65535).astype(np.uint16)
    lum_curved = lut[lum_clipped].astype(np.float32)

    # Calculate scaling factor (avoid division by zero)
    # Use np.divide with where to prevent divide-by-zero warning
    scale = np.ones_like(lum)
    mask = lum > 1.0
    np.divide(lum_curved, lum, out=scale, where=mask)

    # Apply same scale to all channels (preserves hue)
    img *= scale[:, :, np.newaxis]

    # Clamp and convert back to big-endian 16-bit
    img = np.clip(img, 0, 65535).astype(">u2")

    # Write PPM back
    with open(ppm_path, "wb") as f:
        f.write(magic)
        f.write(f"{width} {height}\n".encode())
        f.write(f"{maxval}\n".encode())
        f.write(img.tobytes())


def _adjust_saturation(ppm_path: Path, factor: float) -> None:
    """Adjust saturation of PPM image.

    Uses a simple RGB-based saturation adjustment that scales the distance
    of each channel from the luminance value.

    Formula: output = luminance + (input - luminance) * factor

    Args:
        ppm_path: Path to 16-bit PPM file (modified in place).
        factor: Saturation multiplier (1.0 = no change, 1.2 = +20%, 0.8 = -20%).
    """
    if abs(factor - 1.0) < 0.001:
        return  # No adjustment needed

    # Read PPM header and data
    with open(ppm_path, "rb") as f:
        magic = f.readline()
        if magic.strip() != b"P6":
            raise ValueError(f"Not a binary PPM file: {ppm_path}")

        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        dims = line.decode().split()
        width, height = int(dims[0]), int(dims[1])

        maxval_line = f.readline()
        maxval = int(maxval_line.decode().strip())

        data = f.read()

    if maxval != 65535:
        raise ValueError(f"Expected 16-bit PPM (maxval 65535), got {maxval}")

    # Parse as big-endian 16-bit unsigned integers
    img = np.frombuffer(data, dtype=">u2").reshape(height, width, 3).astype(np.float32)

    # Calculate luminance (BT.709 coefficients)
    luminance = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]

    # Adjust saturation: output = luminance + (input - luminance) * factor
    for c in range(3):
        img[:, :, c] = luminance + (img[:, :, c] - luminance) * factor

    # Clamp and convert back to big-endian 16-bit
    img = np.clip(img, 0, 65535).astype(">u2")

    # Write PPM back
    with open(ppm_path, "wb") as f:
        f.write(magic)
        f.write(f"{width} {height}\n".encode())
        f.write(f"{maxval}\n".encode())
        f.write(img.tobytes())


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


def process_all(
    videos: list[tuple[VideoInfo, Path]],
    images: list[tuple[ImageInfo, Path]],
    max_workers: int,
    preset: int,
    crf: int,
) -> None:
    """Process all media (videos and images) with parallel execution.

    Raises:
        FFmpegError: On video encoding failure.
        JxlExtractionError: On image extraction failure.
    """
    dest_dir = Path(DESTINATION_DIR)

    total = len(videos) + len(images)
    console.print(
        f"\n[bold blue]Processing {len(videos)} video(s) and {len(images)} image(s)[/]"
    )
    if videos:
        console.print(f"  Video: Preset {preset}, CRF {crf}")
    console.print(f"  Workers: {max_workers}\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Track futures with their media info and type
        future_to_media: dict[Future[Path], tuple[str, VideoInfo | ImageInfo]] = {}

        # Submit video conversion tasks
        for info, _ in videos:
            future = executor.submit(convert_video, info, dest_dir, preset, crf)
            future_to_media[future] = ("video", info)

        # Submit image processing tasks
        for info, _ in images:
            future = executor.submit(process_image, info, dest_dir)
            future_to_media[future] = ("image", info)

        completed = 0

        for future in as_completed(future_to_media):
            media_type, info = future_to_media[future]
            completed += 1

            try:
                output_path = future.result()
                console.print(
                    f"[green]✓[/] [{completed}/{total}] "
                    f"{info.path.name} → {output_path.name}"
                )
            except (FFmpegError, JxlExtractionError) as e:
                console.print(f"[red]✗[/] [{completed}/{total}] {e}")
                # Cancel remaining futures
                for f in future_to_media:
                    f.cancel()
                raise


# ═══════════════════════════════════════════════════════════════════
#                        IMAGE PROCESSING
# ═══════════════════════════════════════════════════════════════════


def process_dng_to_jxl(info: ImageInfo, dest_dir: Path) -> Path:
    """Process iPhone ProRAW DNG to JXL with configurable look.

    Processing pipeline varies based on TONE_CURVE_STYLE:

    Phase 1 (dcraw_emu_dng):
    - Decode JXL-compressed Linear Raw data via DNG SDK
    - Demosaic Bayer pattern to RGB using AHD algorithm
    - Apply BaselineExposure correction (for apple/vivid/film styles)
    - Apply color matrix and highlight handling
    - Output to 16-bit PPM in Display P3 color space

    Phase 2 (Python post-processing):
    - Apply tone curve (apple=from DNG, vivid/film=preset, flat/linear=skip/linear)
    - Apply saturation adjustment (vivid +20%, film +5%)

    Styles:
    - apple: Match Apple Photos (DNG ProfileToneCurve)
    - flat: Low contrast for grading (linear curve, -W flag)
    - vivid: High contrast, punchy (aggressive S-curve, +20% saturation)
    - film: Soft contrast, warm (gentle S-curve, +5% saturation)
    - linear: Raw sensor data (-W flag, no curve)

    Raises:
        JxlExtractionError: If processing fails.
    """
    output_path = dest_dir / f"{info.path.stem}.jxl"
    temp_ppm = dest_dir / f".{info.path.stem}_temp.ppm"

    # Use dcraw_emu_dng from script directory
    script_dir = Path(__file__).resolve().parent
    dcraw_emu = script_dir / "dcraw_emu_dng"

    # Determine processing based on tone curve style
    style = TONE_CURVE_STYLE.lower()

    if style == "flat":
        # For flat: apply BaselineExposure + 1 EV boost, use sRGB gamma
        base_exp = _get_baseline_exposure(info.path)
        exp_shift = base_exp * 2.0  # +1 EV additional compensation
        highlight_preserve = 0.0
        highlight_mode = "0"  # Clip highlights
        use_srgb_gamma = True
    elif style == "linear":
        # For linear: raw sensor data, no adjustments
        exp_shift = 1.0
        highlight_preserve = 0.0
        highlight_mode = "0"  # Clip highlights
        use_srgb_gamma = False
    else:
        # For apple/vivid/film: use per-image BaselineExposure
        exp_shift = _get_baseline_exposure(info.path)
        highlight_preserve = 0.5
        highlight_mode = "2"  # Blend highlights
        use_srgb_gamma = False

    try:
        # Build environment with library path for dcraw_emu_dng
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = _get_dcraw_library_path()

        # Phase 1: Process RAW to 16-bit PPM using dcraw_emu_dng with DNG SDK
        dcraw_cmd = [
            str(dcraw_emu),
            "-dngsdk",  # Use DNG SDK to decode JXL-compressed RAW
            "-6",  # 16-bit output
            "-q", "3",  # AHD demosaicing (high quality)
            "-H", highlight_mode,  # Highlight handling
            "-aexpo", f"{exp_shift:.4f}", f"{highlight_preserve:.1f}",
            "-o", "6",  # Output color space: Display P3
            "-Z", str(temp_ppm),
            str(info.path),
        ]

        # Add -W flag for linear/flat styles (disable auto-brightness)
        if style in ("flat", "linear"):
            dcraw_cmd.insert(1, "-W")

        # Add sRGB gamma for flat style (2.4 gamma with 12.92 linear slope)
        if use_srgb_gamma:
            dcraw_cmd.insert(1, "-g")
            dcraw_cmd.insert(2, "2.4")
            dcraw_cmd.insert(3, "12.92")

        result = subprocess.run(dcraw_cmd, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            raise JxlExtractionError(
                f"dcraw_emu_dng failed: {result.stderr or result.stdout}"
            )

        if not temp_ppm.exists():
            raise JxlExtractionError(
                f"dcraw_emu_dng did not produce output file for {info.path.name}"
            )

        # Phase 2: Apply tone curve based on style
        if style == "linear":
            pass  # Skip tone curve entirely for linear output
        else:
            if style == "apple":
                curve_points = _extract_tone_curve(info.path)
            else:
                curve_points = TONE_CURVE_PRESETS.get(style)

            if curve_points and len(curve_points) > 2:
                lut = _build_tone_curve_lut(curve_points)
                _apply_tone_curve_to_ppm(temp_ppm, lut)

        # Phase 3: Apply saturation adjustment
        sat_factor = SATURATION_ADJUSTMENTS.get(style, 1.0)
        if sat_factor != 1.0:
            _adjust_saturation(temp_ppm, sat_factor)

        # Encode 16-bit PPM to lossless JXL
        cjxl_cmd = [
            "cjxl",
            str(temp_ppm),
            str(output_path),
            "-d", "0",  # Lossless (distance 0)
            "-e", "7",  # Encoding effort (1-10)
        ]

        result = subprocess.run(cjxl_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise JxlExtractionError(
                f"cjxl encoding failed: {result.stderr or result.stdout}"
            )

    except JxlExtractionError:
        raise
    except Exception as e:
        if output_path.exists():
            output_path.unlink()
        raise JxlExtractionError(f"Failed to process {info.path.name}: {e}")
    finally:
        # Clean up temp file
        if temp_ppm.exists():
            temp_ppm.unlink()

    # Copy all metadata from DNG to JXL using exiftool
    try:
        subprocess.run(
            [
                "exiftool",
                "-overwrite_original",
                "-TagsFromFile",
                str(info.path),
                "-all:all",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        # Non-fatal: continue even if metadata copy fails
        console.print(
            f"[yellow]Warning:[/] Failed to copy metadata for {info.path.name}"
        )

    # Copy filesystem timestamps (mtime, atime) from source
    shutil.copystat(info.path, output_path)

    return output_path


def copy_image(info: ImageInfo, dest_dir: Path) -> Path:
    """Copy image file preserving all metadata.

    Uses shutil.copy2 which preserves file metadata (mtime, etc.).
    """
    output_path = dest_dir / info.path.name
    shutil.copy2(info.path, output_path)
    return output_path


def process_image(info: ImageInfo, dest_dir: Path) -> Path:
    """Process single image - process DNG to JXL or copy as-is.

    Raises:
        JxlExtractionError: If DNG processing fails.
    """
    if info.should_extract_jxl:
        return process_dng_to_jxl(info, dest_dir)
    else:
        return copy_image(info, dest_dir)


# ═══════════════════════════════════════════════════════════════════
#                        MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════


def main() -> None:
    """Script entry point."""
    console.print("\n[bold]Media Converter[/] (AV1 Video + Image Extraction)\n")

    try:
        # Validate environment
        validate_environment()

        source_dir = Path(SOURCE_DIR)
        dest_dir = Path(DESTINATION_DIR)

        # Scan for videos and images
        video_paths = scan_videos(source_dir)
        image_paths = scan_images(source_dir)

        if not video_paths and not image_paths:
            console.print(f"[yellow]No media files found in {source_dir}[/]")
            return

        console.print(
            f"Found {len(video_paths)} video(s) and {len(image_paths)} image(s) "
            "to analyze...\n"
        )

        # Probe and filter media
        videos_to_process: list[tuple[VideoInfo, Path]] = []
        images_to_process: list[tuple[ImageInfo, Path]] = []
        skipped_videos: list[tuple[str, str]] = []
        skipped_images: list[tuple[str, str]] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            # Analyze videos
            if video_paths:
                task = progress.add_task(
                    "Analyzing videos...", total=len(video_paths)
                )
                for path in video_paths:
                    try:
                        info = probe_video(path)
                        should, reason = should_process(info, dest_dir)

                        if should:
                            output_path = dest_dir / f"{info.path.stem}.mp4"
                            videos_to_process.append((info, output_path))
                        else:
                            skipped_videos.append((path.name, reason))

                    except FFprobeError as e:
                        console.print(f"[red]Error probing {path.name}:[/] {e}")
                        raise

                    progress.advance(task)

            # Analyze images
            if image_paths:
                task = progress.add_task(
                    "Analyzing images...", total=len(image_paths)
                )
                for path in image_paths:
                    try:
                        info = probe_image(path)
                        should, reason = should_process_image(info, dest_dir)

                        if should:
                            if info.should_extract_jxl:
                                output_path = dest_dir / f"{info.path.stem}.jxl"
                            else:
                                output_path = dest_dir / info.path.name
                            images_to_process.append((info, output_path))
                        else:
                            skipped_images.append((path.name, reason))

                    except ExiftoolError as e:
                        console.print(f"[red]Error probing {path.name}:[/] {e}")
                        raise

                    progress.advance(task)

        # Display video summary table
        if videos_to_process or skipped_videos:
            table = Table(title="Video Analysis Results")
            table.add_column("File", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Resolution")
            table.add_column("FPS")
            table.add_column("Codec")
            table.add_column("HDR")
            table.add_column("DV")
            table.add_column("Audio")

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

            for name, reason in skipped_videos:
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

        # Display image summary table
        if images_to_process or skipped_images:
            table = Table(title="Image Analysis Results")
            table.add_column("File", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Dimensions")
            table.add_column("Format")
            table.add_column("Compression")
            table.add_column("Action")

            for info, output_path in images_to_process:
                if info.should_extract_jxl:
                    action = "[blue]Process RAW[/]"
                    status = "[green]To Process[/]"
                else:
                    action = "[dim]Copy[/]"
                    status = "[green]To Copy[/]"

                table.add_row(
                    info.path.name,
                    status,
                    f"{info.width}x{info.height}" if info.width else "-",
                    info.format.upper(),
                    info.compression or "-",
                    action,
                )

            for name, reason in skipped_images:
                table.add_row(
                    name,
                    f"[dim]{reason}[/]",
                    "-",
                    "-",
                    "-",
                    "-",
                )

            console.print(table)

        if not videos_to_process and not images_to_process:
            console.print("\n[yellow]No media to process.[/]")
            return

        # Process all media
        process_all(
            videos_to_process, images_to_process, MAX_WORKERS, PRESET, CRF
        )

        # Summary
        video_count = len(videos_to_process)
        image_count = len(images_to_process)
        jxl_count = sum(1 for info, _ in images_to_process if info.should_extract_jxl)
        copy_count = image_count - jxl_count

        summary_parts = []
        if video_count:
            summary_parts.append(f"{video_count} video(s) converted")
        if jxl_count:
            summary_parts.append(f"{jxl_count} DNG(s) processed to JXL")
        if copy_count:
            summary_parts.append(f"{copy_count} image(s) copied")

        console.print(f"\n[bold green]Done![/] {', '.join(summary_parts)}.")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/]")
        sys.exit(130)
    except FFmpegError as e:
        console.print(f"\n[red]Video encoding failed:[/] {e}")
        sys.exit(1)
    except JxlExtractionError as e:
        console.print(f"\n[red]DNG processing failed:[/] {e}")
        sys.exit(1)
    except FFprobeError as e:
        console.print(f"\n[red]Video analysis failed:[/] {e}")
        sys.exit(1)
    except ExiftoolError as e:
        console.print(f"\n[red]Image analysis failed:[/] {e}")
        sys.exit(1)
    except ValidationError as e:
        console.print(f"\n[red]Configuration error:[/] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
