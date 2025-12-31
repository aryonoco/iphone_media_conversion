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
Photo Converter Script

Processes iPhone 16/17 Pro ProRAW DNG files to HDR JPEG XL.
Copies other image formats (HEIC, HEIF, AVIF, JPEG, PNG) as-is with metadata.

Images: DNG with JPEG XL compression -> processed to HDR .jxl
        (uses dcraw_emu_dng with DNG SDK for full RAW processing)
        Other images and older DNGs -> copied unchanged.

Output format:
    - Color space: Rec.2020 (ITU-R BT.2020)
    - Transfer function: PQ (SMPTE ST 2084)
    - HDR standard: Rec.2100
    - Peak luminance: 10,000 nits

Prerequisites:
    - Run libraw_dng.py first to build dcraw_emu_dng
    - Requires: exiftool, cjxl

Usage:
    1. Edit the CONFIGURATION section below
    2. Run: ./export_photos.py
    Or: uv run export_photos.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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

SOURCE_DIR: str = "/var/home/admin/Pictures/iphone-orig"
DESTINATION_DIR: str = "/var/home/admin/Pictures/script-output"

# Parallel Processing
MAX_WORKERS: int = 6  # Number of images to process simultaneously

# ═══════════════════════════════════════════════════════════════════
#                        END CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

# Constants - Image
SUPPORTED_IMAGE_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".jpg", ".jpeg", ".heic", ".heif", ".avif", ".png", ".dng"}
)

# Image formats that should be copied without processing
COPY_ONLY_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".jpg", ".jpeg", ".heic", ".heif", ".avif", ".png"}
)

# Rich console for output
console = Console()

# Global flags (set by argument parser)
DEBUG_MODE: bool = False
DEBUG_OUTPUT_DIR: Path | None = None
PGTM_STRENGTH: float = 0.5  # 0.5 matches Apple Photos rendering
EXPOSURE_OFFSET: float = 0.0  # Additional exposure offset in EV


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert iPhone ProRAW DNG files to HDR JPEG XL (Rec.2100 PQ).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output format: Rec.2020 color space, PQ transfer function, 10000 nits peak.

Examples:
  %(prog)s                          # Use default source/dest directories
  %(prog)s --source ~/Photos/raw    # Custom source directory
  %(prog)s --debug                  # Output intermediate PPM files for debugging
  %(prog)s --debug --debug-dir /tmp/debug  # Custom debug output directory
""",
    )
    parser.add_argument(
        "--source", "-s",
        type=Path,
        default=Path(SOURCE_DIR),
        help=f"Source directory containing DNG files (default: {SOURCE_DIR})",
    )
    parser.add_argument(
        "--dest", "-d",
        type=Path,
        default=Path(DESTINATION_DIR),
        help=f"Destination directory for JXL output (default: {DESTINATION_DIR})",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode: output intermediate PPM files at each pipeline stage",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="Directory for debug output (default: <dest>/debug/)",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force reprocessing of files that already have output",
    )
    parser.add_argument(
        "--pgtm-strength",
        type=float,
        default=0.5,
        help=(
            "ProfileGainTableMap strength (0.0-1.0). Controls how strongly PGTM "
            "local tone mapping is applied. 0.5 matches Apple Photos (default: 0.5)"
        ),
    )
    parser.add_argument(
        "--exposure-offset",
        type=float,
        default=0.0,
        help=(
            "Additional exposure offset in EV stops. Applied after BaselineExposure. "
            "Use negative values to darken output (default: 0.0)"
        ),
    )
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════
#                        EXCEPTIONS
# ═══════════════════════════════════════════════════════════════════


class ValidationError(Exception):
    """Raised when environment validation fails."""


class ExiftoolError(Exception):
    """Raised when exiftool fails to analyze an image."""


class JxlExtractionError(Exception):
    """Raised when JXL extraction from DNG fails."""


# ═══════════════════════════════════════════════════════════════════
#                        DATA MODELS
# ═══════════════════════════════════════════════════════════════════


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

    # Check cjxl (local build required)
    script_dir = Path(__file__).resolve().parent
    cjxl_path = script_dir / "build" / "deps" / "bin" / "cjxl"
    if not cjxl_path.exists():
        raise ValidationError(
            f"Local cjxl not found at {cjxl_path}. Run libraw_dng.py to build dependencies."
        )
    # Verify it runs with proper library path
    try:
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = _get_library_path()
        subprocess.run(
            [str(cjxl_path), "--version"],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
    except subprocess.CalledProcessError:
        raise ValidationError("cjxl failed to run")

    # Check dcraw_emu_dng (for iPhone ProRAW DNG processing)
    script_dir = Path(__file__).resolve().parent
    dcraw_emu = script_dir / "build" / "dcraw_emu_dng"
    if not dcraw_emu.exists():
        raise ValidationError(
            f"dcraw_emu_dng not found at {dcraw_emu}. Run libraw_dng.py to build it."
        )

    # Test that dcraw_emu_dng can actually run with required libraries
    try:
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = _get_library_path()
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


def _get_library_path() -> str:
    """Build LD_LIBRARY_PATH for local binaries (dcraw_emu_dng, cjxl).

    Libraries:
    - ./build/lib: libraw
    - ./build/deps/lib64: libjxl, libbrotli, etc.
    - Homebrew LLVM: libc++, libomp (clang runtime)

    Returns:
        Colon-separated library path string.
    """
    paths: list[str] = []
    script_dir = Path(__file__).resolve().parent

    # LibRaw libraries
    build_lib = script_dir / "build" / "lib"
    if build_lib.exists():
        paths.append(str(build_lib))

    # libjxl and dependencies
    deps_lib64 = script_dir / "build" / "deps" / "lib64"
    if deps_lib64.exists():
        paths.append(str(deps_lib64))

    # Homebrew LLVM (libc++, libomp) - needed for clang runtime
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


def _get_baseline_exposure_ev(dng_path: Path) -> float:
    """Read BaselineExposure from DNG in EV stops (raw value).

    iPhone ProRAW DNG files contain per-image BaselineExposure values
    (typically 0.02 to 2.36 EV) that indicate how much to brighten the image.

    Returns:
        Exposure value in EV stops (0.0 = no change).
    """
    try:
        result = subprocess.run(
            ["exiftool", "-BaselineExposure", "-n", "-s3", str(dng_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        # Default: no exposure adjustment
        return 0.0


def _extract_profile_tone_curve(dng_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract ProfileToneCurve from DNG for global tone mapping.

    ProfileToneCurve is a 1D lookup table (257 input/output pairs) that applies
    a contrast curve to the image. iPhone ProRAW typically includes an S-curve
    that darkens shadows and adds contrast.

    The curve is stored as 514 float32 values: alternating input, output pairs.
    Input/output values are in [0, 1] range for SDR content.

    Returns:
        Tuple of (input_values, output_values) as numpy arrays, or None if not found.
    """
    import struct

    PROFILE_TONE_CURVE_TAG = 0xC6FC  # 50940

    try:
        with open(dng_path, "rb") as f:
            data = f.read()

        # Check TIFF header
        if data[:2] == b"II":
            fmt = "<"
        elif data[:2] == b"MM":
            fmt = ">"
        else:
            return None

        # Get IFD0 offset
        ifd_offset = struct.unpack(f"{fmt}I", data[4:8])[0]

        def search_ifd(offset: int) -> tuple | None:
            if offset < 2 or offset >= len(data) - 2:
                return None

            num_entries = struct.unpack(f"{fmt}H", data[offset : offset + 2])[0]
            if num_entries > 1000:  # Sanity check
                return None

            for i in range(num_entries):
                tag_offset = offset + 2 + i * 12
                if tag_offset + 12 > len(data):
                    break

                tag_id, tag_type, count = struct.unpack(
                    f"{fmt}HHI", data[tag_offset : tag_offset + 8]
                )

                if tag_id == PROFILE_TONE_CURVE_TAG and tag_type == 11:  # FLOAT
                    value_offset = struct.unpack(
                        f"{fmt}I", data[tag_offset + 8 : tag_offset + 12]
                    )[0]
                    floats = struct.unpack(
                        f"{fmt}{count}f",
                        data[value_offset : value_offset + count * 4],
                    )
                    inputs = np.array(floats[::2], dtype=np.float32)
                    outputs = np.array(floats[1::2], dtype=np.float32)
                    return inputs, outputs

                # Check SubIFDs
                if tag_id in [0x14A, 0x8769]:
                    sub_offset = struct.unpack(
                        f"{fmt}I", data[tag_offset + 8 : tag_offset + 12]
                    )[0]
                    result = search_ifd(sub_offset)
                    if result:
                        return result

            # Check next IFD
            next_offset = offset + 2 + num_entries * 12
            if next_offset + 4 <= len(data):
                next_ifd = struct.unpack(
                    f"{fmt}I", data[next_offset : next_offset + 4]
                )[0]
                if next_ifd > 0:
                    return search_ifd(next_ifd)

            return None

        return search_ifd(ifd_offset)

    except Exception:
        return None


def _apply_profile_tone_curve(
    img: np.ndarray,
    curve_inputs: np.ndarray,
    curve_outputs: np.ndarray,
) -> np.ndarray:
    """Apply ProfileToneCurve using DNG SDK's RGBTone algorithm.

    This implements the hue-preserving tone curve from DNG SDK (dng_reference.cpp).
    The algorithm:
    1. For each pixel, identify max, mid, min RGB values
    2. Apply curve to max and min independently
    3. Interpolate mid: mid_out = min_out + (max_out - min_out) * (mid - min) / (max - min)

    This preserves hue while applying proper contrast to each channel.

    For HDR content (values > 1.0), the curve is extended linearly using
    the slope at the endpoint to preserve HDR headroom.

    Args:
        img: Linear RGB image (H, W, 3), may contain values > 1.0 for HDR.
        curve_inputs: Input values for the tone curve (0-1 range).
        curve_outputs: Output values for the tone curve.

    Returns:
        Tone-mapped image with curve applied, hue preserved.
    """
    # Create interpolator for smooth curve
    interp = PchipInterpolator(curve_inputs, curve_outputs, extrapolate=False)

    # Get endpoint slope for HDR extension
    endpoint_slope = (curve_outputs[-1] - curve_outputs[-2]) / (
        curve_inputs[-1] - curve_inputs[-2] + 1e-10
    )

    def apply_curve(values: np.ndarray) -> np.ndarray:
        """Apply tone curve with HDR linear extension."""
        result = np.zeros_like(values)
        # SDR range: use interpolated curve
        sdr_mask = (values >= 0) & (values <= 1.0)
        result[sdr_mask] = interp(values[sdr_mask])
        # HDR range: linear extension
        hdr_mask = values > 1.0
        result[hdr_mask] = curve_outputs[-1] + endpoint_slope * (values[hdr_mask] - 1.0)
        # Negative (shouldn't happen but be safe)
        result[values < 0] = 0.0
        return result

    h, w = img.shape[:2]
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # Find max, mid, min for each pixel
    rgb_stack = np.stack([r, g, b], axis=-1)
    max_val = np.max(rgb_stack, axis=-1)
    min_val = np.min(rgb_stack, axis=-1)

    # Apply curve to max and min
    max_curved = apply_curve(max_val)
    min_curved = apply_curve(min_val)

    # For mid values, interpolate based on position between min and max
    # mid_out = min_out + (max_out - min_out) * (mid - min) / (max - min)
    result = np.zeros_like(img)

    # Handle case where max == min (grayscale pixels)
    gray_mask = (max_val - min_val) < 1e-10

    for i, channel in enumerate([r, g, b]):
        # Compute normalized position of this channel between min and max
        # t = (channel - min) / (max - min)
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.where(gray_mask, 0.5, (channel - min_val) / (max_val - min_val + 1e-10))
            t = np.clip(t, 0.0, 1.0)

        # Interpolate: out = min_curved + t * (max_curved - min_curved)
        result[:, :, i] = min_curved + t * (max_curved - min_curved)

    return result.astype(np.float32)


def _extract_profile_gain_table_map(dng_path: Path) -> dict | None:
    """Extract ProfileGainTableMap (PGTM) from DNG for local tone mapping.

    DNG 1.6 introduces ProfileGainTableMap - a 3D lookup table for spatially-varying
    local tone mapping. iPhone ProRAW uses ~197KB PGTM data per image to create the
    characteristic "Apple look" with lifted shadows and controlled highlights.

    Binary format (DNG 1.6 spec / dng_gain_map.cpp):
    - uint32 MapPointsV: Grid vertical points
    - uint32 MapPointsH: Grid horizontal points
    - float64 MapSpacingV, MapSpacingH: Grid spacing
    - float64 MapOriginV, MapOriginH: Grid origin
    - uint32 NumTablePoints: Points in weight dimension
    - float32[5] MapInputWeights: R, G, B, min(RGB), max(RGB) weights
    - float32[V*H*N] Table: 3D gain values

    Args:
        dng_path: Path to DNG file.

    Returns:
        Dictionary with PGTM parameters and table, or None if not present.
    """
    import struct

    try:
        # Extract binary ProfileGainTableMap data using exiftool
        result = subprocess.run(
            ["exiftool", "-b", "-ProfileGainTableMap", str(dng_path)],
            capture_output=True,
            check=True,
        )
        data = result.stdout
        if len(data) < 64:  # Minimum header size
            return None

        # Parse header (BIG-ENDIAN format) per DNG SDK dng_gain_map.cpp
        # Header: V(4) H(4) spacingV(8) spacingH(8) originV(8) originH(8) N(4) weights(20)
        points_v = struct.unpack_from(">I", data, 0)[0]
        points_h = struct.unpack_from(">I", data, 4)[0]
        spacing_v = struct.unpack_from(">d", data, 8)[0]
        spacing_h = struct.unpack_from(">d", data, 16)[0]
        origin_v = struct.unpack_from(">d", data, 24)[0]
        origin_h = struct.unpack_from(">d", data, 32)[0]
        num_table_points = struct.unpack_from(">I", data, 40)[0]
        input_weights = struct.unpack_from(">5f", data, 44)  # R, G, B, min, max

        # Table starts at offset 64: float32[V * H * N] in big-endian
        table_size = points_v * points_h * num_table_points
        expected_data_len = 64 + table_size * 4
        if len(data) < expected_data_len:
            return None

        # Parse big-endian float32 table
        table = np.frombuffer(data, dtype=">f4", count=table_size, offset=64)
        table = table.reshape(points_v, points_h, num_table_points)

        return {
            "points_v": points_v,
            "points_h": points_h,
            "spacing_v": spacing_v,
            "spacing_h": spacing_h,
            "origin_v": origin_v,
            "origin_h": origin_h,
            "num_table_points": num_table_points,
            "input_weights": input_weights,  # (R, G, B, min, max)
            "table": table,  # Shape: (V, H, N)
        }
    except (subprocess.CalledProcessError, ValueError):
        return None


def _compute_pgtm_gains(
    img: np.ndarray,
    pgtm: dict[str, Any],
    baseline_exposure_ev: float = 0.0,
) -> np.ndarray:
    """Compute PGTM gains for a numpy array without applying or clipping.

    This function extracts the gain computation from _apply_profile_gain_table_map
    for use in HDR pipelines where we need gains without clipping.

    Per DNG SDK (dng_reference.cpp:3375-3385), the weight is computed from
    RAW pixel values and then scaled by 2^BaselineExposure before table lookup.

    Args:
        img: Float32 array shape (H, W, 3), RAW linear values (before exposure).
        pgtm: Dictionary from _extract_profile_gain_table_map().
        baseline_exposure_ev: BaselineExposure in EV stops. Weight is scaled
            by 2^baseline_exposure_ev per DNG SDK.

    Returns:
        2D float32 array of scaled gains, shape (H, W).
    """
    from scipy.interpolate import RegularGridInterpolator

    height, width = img.shape[:2]

    # Extract PGTM parameters
    points_v = pgtm["points_v"]
    points_h = pgtm["points_h"]
    num_table_points = pgtm["num_table_points"]
    spacing_v = pgtm["spacing_v"]
    spacing_h = pgtm["spacing_h"]
    origin_v = pgtm["origin_v"]
    origin_h = pgtm["origin_h"]
    input_weights = np.array(pgtm["input_weights"])
    table = pgtm["table"]

    # Calculate weight per pixel (clamp to 0-1 for table lookup)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    rgb_min = np.minimum(np.minimum(r, g), b)
    rgb_max = np.maximum(np.maximum(r, g), b)

    weight = (
        input_weights[0] * r
        + input_weights[1] * g
        + input_weights[2] * b
        + input_weights[3] * rgb_min
        + input_weights[4] * rgb_max
    )
    # DNG SDK scales weight by 2^BaselineExposure, but this can cause weight
    # saturation for extreme exposure images. We cap the scaling factor to
    # prevent complete saturation while still adjusting for exposure level.
    MAX_WEIGHT_SCALE = 4.0  # Cap at 2 EV of adjustment
    if baseline_exposure_ev != 0.0:
        exposure_weight_gain = 2.0 ** baseline_exposure_ev
        # Cap the scaling to prevent saturation
        exposure_weight_gain = min(exposure_weight_gain, MAX_WEIGHT_SCALE)
        weight = weight * exposure_weight_gain
    # Clamp weight to valid table range (but don't clamp the pixel values)
    weight = np.clip(weight, 0.0, 1.0)

    # Map weight to table index per DNG SDK (dng_reference.cpp:3399):
    # weightScaled = weight * tableSize (NOT tableSize-1)
    weight_idx = weight * num_table_points
    weight_idx = np.clip(weight_idx, 0, num_table_points - 1)

    # Compute spatial coordinates
    rows = np.arange(height, dtype=np.float32) + 0.5
    cols = np.arange(width, dtype=np.float32) + 0.5
    v_image = rows / height
    u_image = cols / width
    row_coords = np.clip((v_image - origin_v) / spacing_v, 0, points_v - 1)
    col_coords = np.clip((u_image - origin_h) / spacing_h, 0, points_h - 1)
    row_grid, col_grid = np.meshgrid(row_coords, col_coords, indexing="ij")

    # Build interpolator
    v_axis = np.arange(points_v)
    h_axis = np.arange(points_h)
    n_axis = np.arange(num_table_points)
    interp = RegularGridInterpolator(
        (v_axis, h_axis, n_axis), table, method="linear",
        bounds_error=False, fill_value=1.0,
    )

    # Interpolate gains
    coords = np.stack([row_grid.ravel(), col_grid.ravel(), weight_idx.ravel()], axis=1)
    gains = interp(coords).reshape(height, width)

    # Scale by PGTM_STRENGTH
    scaled_gains = 1.0 + (gains - 1.0) * PGTM_STRENGTH

    return scaled_gains


def _convert_prophoto_to_rec2020(img: np.ndarray) -> np.ndarray:
    """Convert linear ProPhoto RGB to linear Rec.2020 with gamut mapping.

    ProPhoto RGB (RIMM/ROMM) is the DNG SDK's working color space.
    This function converts to Rec.2020 for output encoding.

    The matrix is derived by:
    1. ProPhoto RGB → XYZ (D50 white point)
    2. Bradford chromatic adaptation D50 → D65
    3. XYZ → Rec.2020 (D65 white point)

    For out-of-gamut colors (negative RGB values after conversion), uses
    luminance-preserving desaturation toward neutral to maintain hue.

    Args:
        img: Linear ProPhoto RGB image as float32 array (H, W, 3)

    Returns:
        Linear Rec.2020 image as float32 array (H, W, 3)
    """
    # Pre-computed ProPhoto RGB → Rec.2020 matrix
    # Via: ProPhoto→XYZ(D50) → Bradford(D50→D65) → XYZ→Rec.2020(D65)
    # Verified: neutral gray (0.5, 0.5, 0.5) maps to (0.5, 0.5, 0.5)
    matrix = np.array([
        [ 1.2006765, -0.0574641, -0.1431306],
        [-0.0699240,  1.0805933, -0.0106823],
        [ 0.0055355, -0.0407676,  1.0350179],
    ], dtype=np.float32)

    # Rec.2020 luminance coefficients (ITU-R BT.2020)
    lum_coeffs = np.array([0.2627, 0.6780, 0.0593], dtype=np.float32)

    h, w, c = img.shape
    flat = img.reshape(-1, 3)
    converted = flat @ matrix.T

    # Find pixels with negative channels (out of Rec.2020 gamut)
    min_vals = np.min(converted, axis=1)
    out_of_gamut = min_vals < 0

    if np.any(out_of_gamut):
        # Calculate luminance for out-of-gamut pixels
        lum = (converted[out_of_gamut] @ lum_coeffs)[:, np.newaxis]

        # Desaturate toward neutral until in gamut
        # Solve for t: lum + t*(rgb - lum) >= 0 for all channels
        rgb_oog = converted[out_of_gamut]

        # For each channel, find t that brings it to 0
        with np.errstate(divide="ignore", invalid="ignore"):
            diff = lum - rgb_oog
            t = np.where(diff > 0, lum / diff, 1.0)
            t_max = np.min(np.where(rgb_oog < 0, t, 1.0), axis=1, keepdims=True)

        # Apply desaturation
        converted[out_of_gamut] = lum + t_max * (rgb_oog - lum)

        # Clean up floating point errors
        converted[out_of_gamut] = np.maximum(converted[out_of_gamut], 0.0)

    return converted.reshape(h, w, 3).astype(np.float32)


def _apply_pq_oetf(linear_nits: np.ndarray) -> np.ndarray:
    """Apply PQ (SMPTE ST 2084) OETF.

    Encodes absolute luminance (0-10000 nits) to PQ signal (0-1).
    This is the standard transfer function for HDR content per ITU-R BT.2100.

    Args:
        linear_nits: Linear light values in nits (0-10000 range)

    Returns:
        PQ-encoded values (0-1 range)
    """
    # PQ constants per SMPTE ST 2084
    m1 = 0.1593017578125    # 2610 / 16384
    m2 = 78.84375           # 2523 / 32 * 128
    c1 = 0.8359375          # 3424 / 4096
    c2 = 18.8515625         # 2413 / 128
    c3 = 18.6875            # 2392 / 128

    # Normalize to 0-1 range (10000 nits = 1.0)
    L = np.clip(linear_nits / 10000.0, 0.0, 1.0)

    # PQ OETF: E' = ((c1 + c2*L^m1) / (1 + c3*L^m1))^m2
    L_m1 = np.power(L, m1)
    numerator = c1 + c2 * L_m1
    denominator = 1.0 + c3 * L_m1
    pq = np.power(numerator / denominator, m2)

    return pq.astype(np.float32)


def _scale_to_nits(linear_normalized: np.ndarray, sdr_white_nits: float = 203.0) -> np.ndarray:
    """Scale normalized linear values to absolute nits for PQ encoding.

    Per ITU-R BT.2408, SDR reference white = 203 nits.
    Scene-referred values can exceed 1.0 for HDR highlights,
    which will map to values above 203 nits.

    Args:
        linear_normalized: Linear values where 1.0 = SDR white
        sdr_white_nits: SDR reference white in nits (default: 203 per BT.2408)

    Returns:
        Linear values in nits (0 to 10000+ range)
    """
    return linear_normalized * sdr_white_nits


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


def should_process_image(
    info: ImageInfo, dest_dir: Path, force: bool = False
) -> tuple[bool, str]:
    """Determine if image should be processed.

    Args:
        info: Image information from probe_image()
        dest_dir: Destination directory
        force: If True, reprocess even if output exists

    Returns:
        Tuple of (should_process, reason_if_skipped)
    """
    if info.should_extract_jxl:
        output_path = dest_dir / f"{info.path.stem}.jxl"
    else:
        output_path = dest_dir / info.path.name

    if output_path.exists() and not force:
        return False, "Output exists"

    return True, ""


# ═══════════════════════════════════════════════════════════════════
#                        IMAGE PROCESSING
# ═══════════════════════════════════════════════════════════════════


def _read_ppm_as_float(ppm_path: Path) -> tuple[np.ndarray, int, int]:
    """Read 16-bit PPM as normalized float32 array.

    Returns:
        Tuple of (image array normalized 0-1, width, height)
    """
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
        if maxval != 65535:
            raise ValueError(f"Expected 16-bit PPM (maxval 65535), got {maxval}")
        data = f.read()
    img = np.frombuffer(data, dtype=">u2").reshape(height, width, 3).astype(np.float32)
    return img / 65535.0, width, height


def _write_ppm_from_float(ppm_path: Path, img: np.ndarray, width: int, height: int) -> None:
    """Write normalized float32 array as 16-bit PPM."""
    img_out = np.clip(img * 65535.0, 0, 65535).astype(">u2")
    with open(ppm_path, "wb") as f:
        f.write(b"P6\n")
        f.write(f"{width} {height}\n".encode())
        f.write(b"65535\n")
        f.write(img_out.tobytes())


def _save_debug_stage(
    ppm_path: Path,
    stage_name: str,
    image_stem: str,
    metadata: dict | None = None,
) -> None:
    """Save intermediate PPM and statistics for debugging.

    Copies the PPM file to the debug directory and computes luminance statistics.
    Also writes a JSON file with metadata and statistics for each stage.

    Args:
        ppm_path: Path to the current PPM file
        stage_name: Name of the pipeline stage (e.g., "01_dcraw", "02_exposure")
        image_stem: Image filename stem for output naming
        metadata: Optional additional metadata to include
    """
    if not DEBUG_MODE or DEBUG_OUTPUT_DIR is None:
        return

    # Read PPM data for statistics
    try:
        img, width, height = _read_ppm_as_float(ppm_path)
    except Exception as e:
        console.print(f"[red]Debug: Failed to read PPM for {stage_name}: {e}[/]")
        return

    # Compute luminance using Rec.2020 coefficients
    lum = 0.2627 * img[:, :, 0] + 0.6780 * img[:, :, 1] + 0.0593 * img[:, :, 2]

    stats = {
        "stage": stage_name,
        "image": image_stem,
        "dimensions": f"{width}x{height}",
        "min": float(img.min()),
        "max": float(img.max()),
        "mean": float(img.mean()),
        "luminance": {
            "min": float(lum.min()),
            "max": float(lum.max()),
            "mean": float(lum.mean()),
            "percentiles": {
                "1": float(np.percentile(lum, 1)),
                "5": float(np.percentile(lum, 5)),
                "25": float(np.percentile(lum, 25)),
                "50": float(np.percentile(lum, 50)),
                "75": float(np.percentile(lum, 75)),
                "95": float(np.percentile(lum, 95)),
                "99": float(np.percentile(lum, 99)),
            },
        },
    }

    if metadata:
        stats["metadata"] = metadata

    # Copy PPM to debug directory
    debug_ppm = DEBUG_OUTPUT_DIR / f"{image_stem}_{stage_name}.ppm"
    shutil.copy(ppm_path, debug_ppm)

    # Write statistics JSON
    debug_json = DEBUG_OUTPUT_DIR / f"{image_stem}_{stage_name}.json"
    with open(debug_json, "w") as f:
        json.dump(stats, f, indent=2)

    console.print(f"[dim]Debug: {stage_name} saved (1st%={stats['luminance']['percentiles']['1']:.6f})[/]")


def process_dng_to_jxl(info: ImageInfo, dest_dir: Path) -> Path:
    """Process iPhone ProRAW DNG to HDR JPEG XL with PQ transfer function.

    Implements the DNG SDK rendering pipeline for true HDR output:

    Phase 1 (dcraw_emu_dng):
    - Decode JXL-compressed Linear Raw data via DNG SDK
    - Demosaic Bayer pattern to RGB using AHD algorithm
    - Output 16-bit linear ProPhoto RGB PPM

    Phase 2 (ProfileGainTableMap):
    - Apply DNG 1.6 local tone mapping for spatially-varying adjustments
    - Preserves full HDR dynamic range (no clipping)

    Phase 3 (Exposure):
    - Apply BaselineExposure correction after PGTM (DNG SDK order)
    - Scene-referred data preserved without clipping

    Phase 3.5 (ProfileToneCurve):
    - Apply global S-curve for contrast (darkens shadows)
    - HDR values > 1.0 extended linearly to preserve headroom

    Phase 4 (HDR Output):
    - Convert ProPhoto RGB to Rec.2020
    - Scale to absolute nits (SDR white = 203 nits per BT.2408)
    - Apply PQ (SMPTE ST 2084) transfer function
    - Encode as lossless JXL with Rec2100PQ color space

    Output format:
    - Color space: Rec.2020 (ITU-R BT.2020)
    - Transfer function: PQ (SMPTE ST 2084)
    - HDR standard: Rec.2100
    - Peak luminance: 10,000 nits

    Raises:
        JxlExtractionError: If processing fails.
    """
    output_path = dest_dir / f"{info.path.stem}.jxl"
    temp_ppm = dest_dir / f".{info.path.stem}_temp.ppm"
    temp_hdr_ppm = dest_dir / f".{info.path.stem}_hdr.ppm"

    # Use dcraw_emu_dng from build directory
    script_dir = Path(__file__).resolve().parent
    dcraw_emu = script_dir / "build" / "dcraw_emu_dng"

    # Get baseline exposure for PGTM weight scaling and exposure ramp
    baseline_exposure_ev = _get_baseline_exposure_ev(info.path)

    try:
        # Build environment with library path for dcraw_emu_dng
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = _get_library_path()

        # Phase 1: Process RAW to 16-bit linear PPM using dcraw_emu_dng
        dcraw_cmd = [
            str(dcraw_emu),
            "-W",        # Disable auto-brightness
            "-g", "1", "1",  # Linear output (PQ applied in Python)
            "-dngsdk",   # Use DNG SDK to decode JXL-compressed RAW
            "-6",        # 16-bit output
            "-q", "3",   # AHD demosaicing (high quality)
            "-H", "0",   # Clip highlights at sensor max
            "-o", "4",   # Output color space: ProPhoto RGB (DNG SDK working space)
            "-Z", str(temp_ppm),
            str(info.path),
        ]

        result = subprocess.run(dcraw_cmd, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            raise JxlExtractionError(
                f"dcraw_emu_dng failed: {result.stderr or result.stdout}"
            )

        if not temp_ppm.exists():
            raise JxlExtractionError(
                f"dcraw_emu_dng did not produce output file for {info.path.name}"
            )

        # Debug: Save raw dcraw output
        _save_debug_stage(temp_ppm, "01_dcraw", info.path.stem)

        # Read linear RAW data
        dcraw_linear, width, height = _read_ppm_as_float(temp_ppm)

        # Extract PGTM for local tone mapping
        pgtm = _extract_profile_gain_table_map(info.path)

        # Compute exposure multiplier (BaselineExposure + optional offset)
        total_exposure_ev = baseline_exposure_ev + EXPOSURE_OFFSET
        exposure_mult = 2.0 ** total_exposure_ev

        # Phase 2: Apply ProfileGainTableMap (local tone mapping)
        # Per DNG SDK (dng_reference.cpp:3375-3404):
        # 1. Compute weight from RAW pixels
        # 2. Scale weight by 2^BaselineExposure
        # 3. Look up gain from table
        # 4. Apply gain to pixels
        if pgtm:
            pgtm_gains = _compute_pgtm_gains(dcraw_linear, pgtm, baseline_exposure_ev)
            hdr_linear = dcraw_linear * pgtm_gains[:, :, np.newaxis]
        else:
            hdr_linear = dcraw_linear.copy()

        # Debug: Save post-PGTM output
        _write_ppm_from_float(temp_ppm, hdr_linear, width, height)
        _save_debug_stage(
            temp_ppm, "02_pgtm", info.path.stem,
            {"pgtm_applied": pgtm is not None, "weight_scale": exposure_mult}
        )

        # Phase 3: Apply exposure AFTER PGTM (DNG SDK order)
        hdr_linear = hdr_linear * exposure_mult

        # Debug: Save post-exposure output
        _write_ppm_from_float(temp_ppm, np.clip(hdr_linear, 0, 1), width, height)
        _save_debug_stage(
            temp_ppm, "03_exposure", info.path.stem,
            {"baseline_exposure_ev": baseline_exposure_ev, "multiplier": exposure_mult}
        )

        # Phase 3.5: Apply ProfileToneCurve (global contrast)
        # This S-curve adds characteristic contrast by darkening shadows
        # HDR values > 1.0 are extended linearly to preserve headroom
        tone_curve = _extract_profile_tone_curve(info.path)
        if tone_curve:
            curve_inputs, curve_outputs = tone_curve
            hdr_linear = _apply_profile_tone_curve(hdr_linear, curve_inputs, curve_outputs)

            # Debug: Save post-tone-curve output
            _write_ppm_from_float(temp_ppm, np.clip(hdr_linear, 0, 1), width, height)
            _save_debug_stage(
                temp_ppm, "04_tonecurve", info.path.stem,
                {"curve_points": len(curve_inputs)}
            )

        # Phase 4: Convert to Rec.2020 and apply PQ
        # Convert from ProPhoto RGB to Rec.2020 (still linear)
        hdr_rec2020 = _convert_prophoto_to_rec2020(hdr_linear)

        # Scale to absolute nits (SDR white = 203 nits per ITU-R BT.2408)
        hdr_nits = _scale_to_nits(hdr_rec2020, sdr_white_nits=203.0)

        # Apply PQ transfer function (SMPTE ST 2084)
        hdr_pq = _apply_pq_oetf(hdr_nits)

        # Debug: Save PQ-encoded output
        _write_ppm_from_float(temp_ppm, hdr_pq, width, height)
        _save_debug_stage(
            temp_ppm, "05_pq", info.path.stem,
            {"transfer_function": "PQ", "sdr_white_nits": 203.0}
        )

        # Write final HDR PQ PPM
        _write_ppm_from_float(temp_hdr_ppm, hdr_pq, width, height)

        # Get local cjxl path and environment
        cjxl_path = script_dir / "build" / "deps" / "bin" / "cjxl"
        cjxl_env = os.environ.copy()
        cjxl_env["LD_LIBRARY_PATH"] = _get_library_path()

        # Encode as HDR JXL with Rec2100PQ color space
        cjxl_cmd = [
            str(cjxl_path),
            str(temp_hdr_ppm),
            str(output_path),
            "-d", "0",                          # Lossless compression
            "-e", "7",                          # Encoding effort
            "-x", "color_space=Rec2100PQ",      # Rec.2020 + PQ transfer
            "--intensity_target=10000",          # 10,000 nits peak
        ]

        result = subprocess.run(cjxl_cmd, capture_output=True, text=True, env=cjxl_env)
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
        # Clean up temp files
        for temp_file in [temp_ppm, temp_hdr_ppm]:
            if temp_file.exists():
                temp_file.unlink()

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


def process_all(
    images: list[tuple[ImageInfo, Path]],
    max_workers: int,
    dest_dir: Path,
) -> None:
    """Process all images with parallel execution.

    Raises:
        JxlExtractionError: On image extraction failure.
    """

    total = len(images)
    console.print(f"\n[bold blue]Processing {total} image(s)[/]")
    console.print(f"  Workers: {max_workers}\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Track futures with their image info
        future_to_image: dict[Future[Path], ImageInfo] = {}

        # Submit image processing tasks
        for info, _ in images:
            future = executor.submit(process_image, info, dest_dir)
            future_to_image[future] = info

        completed = 0

        for future in as_completed(future_to_image):
            info = future_to_image[future]
            completed += 1

            try:
                output_path = future.result()
                console.print(
                    f"[green]\u2713[/] [{completed}/{total}] "
                    f"{info.path.name} \u2192 {output_path.name}"
                )
            except JxlExtractionError as e:
                console.print(f"[red]\u2717[/] [{completed}/{total}] {e}")
                # Cancel remaining futures
                for f in future_to_image:
                    f.cancel()
                raise


# ═══════════════════════════════════════════════════════════════════
#                        MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════


def main() -> None:
    """Script entry point."""
    global DEBUG_MODE, DEBUG_OUTPUT_DIR, PGTM_STRENGTH, EXPOSURE_OFFSET

    args = parse_arguments()

    # Set global flags
    DEBUG_MODE = args.debug
    PGTM_STRENGTH = args.pgtm_strength
    EXPOSURE_OFFSET = args.exposure_offset
    if DEBUG_MODE:
        DEBUG_OUTPUT_DIR = args.debug_dir or (args.dest / "debug")
        DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        console.print(f"[yellow]Debug mode enabled. Output: {DEBUG_OUTPUT_DIR}[/]\n")

    console.print("\n[bold]Photo Converter[/] (DNG to JXL + Image Copy)\n")

    try:
        # Validate environment
        validate_environment()

        source_dir = args.source
        dest_dir = args.dest

        # Scan for images
        image_paths = scan_images(source_dir)

        if not image_paths:
            console.print(f"[yellow]No image files found in {source_dir}[/]")
            return

        console.print(f"Found {len(image_paths)} image(s) to analyze...\n")

        # Probe and filter images
        images_to_process: list[tuple[ImageInfo, Path]] = []
        skipped_images: list[tuple[str, str]] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing images...", total=len(image_paths))

            for path in image_paths:
                try:
                    info = probe_image(path)
                    should, reason = should_process_image(info, dest_dir, args.force)

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

        # Display image summary table
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

        if not images_to_process:
            console.print("\n[yellow]No images to process.[/]")
            return

        # Process all images
        process_all(images_to_process, MAX_WORKERS, dest_dir)

        # Summary
        jxl_count = sum(1 for info, _ in images_to_process if info.should_extract_jxl)
        copy_count = len(images_to_process) - jxl_count

        summary_parts = []
        if jxl_count:
            summary_parts.append(f"{jxl_count} DNG(s) processed to JXL")
        if copy_count:
            summary_parts.append(f"{copy_count} image(s) copied")

        console.print(f"\n[bold green]Done![/] {', '.join(summary_parts)}.")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/]")
        sys.exit(130)
    except JxlExtractionError as e:
        console.print(f"\n[red]DNG processing failed:[/] {e}")
        sys.exit(1)
    except ExiftoolError as e:
        console.print(f"\n[red]Image analysis failed:[/] {e}")
        sys.exit(1)
    except ValidationError as e:
        console.print(f"\n[red]Configuration error:[/] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
