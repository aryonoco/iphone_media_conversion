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

Processes iPhone 16/17 Pro ProRAW DNG files to lossless JPEG XL.
Copies other image formats (HEIC, HEIF, AVIF, JPEG, PNG) as-is with metadata.

Images: DNG with JPEG XL compression -> processed to lossless .jxl
        (uses dcraw_emu_dng with DNG SDK for full RAW processing)
        Other images and older DNGs -> copied unchanged.

Prerequisites:
    - Run libraw_dng.py first to build dcraw_emu_dng
    - Requires: exiftool, cjxl

Usage:
    1. Edit the CONFIGURATION section below
    2. Run: ./export_photos.py
    Or: uv run export_photos.py
"""

from __future__ import annotations

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

# Tone Curve Style for DNG Processing
# Controls the visual "look" of converted ProRAW images
TONE_CURVE_STYLE: str = "apple"  # "apple", "flat", "vivid", "film", "linear"

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

# Tone curve presets for different looks
# Format: list of (input, output) control points, normalized 0-1
# None = extract from DNG (apple) or skip curve entirely (linear)
TONE_CURVE_PRESETS: Final[dict[str, list[tuple[float, float]] | None]] = {
    "apple": None,  # Extract ProfileToneCurve from DNG metadata
    "flat": [(0.0, 0.0), (1.0, 1.0)],  # Linear passthrough
    "vivid": [
        # Velvia-inspired: aggressive S-curve, crushed shadows, punchy contrast
        (0.00, 0.00),
        (0.05, 0.02),  # Crushed shadows
        (0.15, 0.10),  # Deep toe
        (0.30, 0.28),  # Shadow contrast
        (0.50, 0.55),  # Pushed midtones (+10%)
        (0.70, 0.78),  # Highlight boost
        (0.85, 0.92),  # Shoulder
        (1.00, 1.00),
    ],
    "film": [
        # Portra-inspired: lifted blacks, soft contrast, compressed highlights
        (0.00, 0.05),  # Significantly lifted blacks
        (0.08, 0.12),  # Shadow lift
        (0.20, 0.25),  # Gentle toe
        (0.40, 0.45),  # Lower-mid
        (0.60, 0.62),  # Upper-mid
        (0.80, 0.78),  # Gentle shoulder
        (0.95, 0.88),  # Compressed highlights
        (1.00, 0.92),  # Rolled-off whites
    ],
    "linear": None,  # No curve applied (with -W flag)
}

# Saturation adjustments per style (1.0 = no change)
SATURATION_ADJUSTMENTS: Final[dict[str, float]] = {
    "apple": 1.0,  # No adjustment - DNG has no HueSatMap data
    "flat": 1.0,  # Neutral
    "vivid": 1.20,  # +20% (Velvia is highly saturated)
    "film": 0.88,  # -12% (Portra is notably desaturated)
    "linear": 1.0,
}

# Display P3 / DCI-P3 luminance coefficients (Y row of RGB-to-XYZ matrix)
# These differ from BT.709/sRGB (0.2126, 0.7152, 0.0722) due to different primaries
# Source: DNG SDK dng_color_space.cpp and standard P3 colorimetry
P3_LUMA_R: Final[float] = 0.2290
P3_LUMA_G: Final[float] = 0.6917
P3_LUMA_B: Final[float] = 0.0793

# Rich console for output
console = Console()


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
            "cjxl not found in PATH. Install libjxl: https://github.com/libjxl/libjxl"
        )
    except subprocess.CalledProcessError:
        raise ValidationError("cjxl failed to run")

    # Check dcraw_emu_dng (for iPhone ProRAW DNG processing)
    script_dir = Path(__file__).resolve().parent
    dcraw_emu = script_dir / "dcraw_emu_dng"
    if not dcraw_emu.exists():
        raise ValidationError(
            f"dcraw_emu_dng not found at {dcraw_emu}. Run libraw_dng.py to build it."
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
        return 2**baseline_ev
    except (subprocess.CalledProcessError, ValueError):
        # Default: no exposure adjustment
        return 1.0


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
                    return [
                        (values[i], values[i + 1]) for i in range(0, len(values), 2)
                    ]

        return [(0.0, 0.0), (1.0, 1.0)]  # Linear fallback
    except (subprocess.CalledProcessError, ValueError, IndexError):
        return [(0.0, 0.0), (1.0, 1.0)]  # Linear fallback


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


def _apply_profile_gain_table_map(
    ppm_path: Path,
    pgtm: dict,
    baseline_exposure: float = 1.0,
) -> None:
    """Apply ProfileGainTableMap local tone mapping to PPM image.

    Implements DNG SDK algorithm from dng_reference.cpp lines 3260-3460:
    1. Calculate weight = sum(input_weights[i] * component[i]) for each pixel
       Components: R, G, B, min(R,G,B), max(R,G,B)
    2. Scale weight by BaselineExposure
    3. Interpolate gain from 3D table using (row, col, weight)
    4. Multiply pixel RGB by gain

    This creates spatially-varying local tone mapping - different regions
    of the image get different tonal adjustments.

    Args:
        ppm_path: Path to 16-bit PPM file (modified in place).
        pgtm: Dictionary from _extract_profile_gain_table_map().
        baseline_exposure: Linear exposure multiplier (2^BaselineExposureEV).
    """
    from scipy.interpolate import RegularGridInterpolator

    # Read PPM
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

    # Parse image data
    img = np.frombuffer(data, dtype=">u2").reshape(height, width, 3).astype(np.float32)

    # Normalize to 0-1 range for weight calculation
    img_norm = img / 65535.0

    # Extract PGTM parameters
    points_v = pgtm["points_v"]
    points_h = pgtm["points_h"]
    num_table_points = pgtm["num_table_points"]
    spacing_v = pgtm["spacing_v"]
    spacing_h = pgtm["spacing_h"]
    origin_v = pgtm["origin_v"]
    origin_h = pgtm["origin_h"]
    input_weights = np.array(pgtm["input_weights"])  # (R, G, B, min, max)
    table = pgtm["table"]  # Shape: (V, H, N)

    # Calculate weight per pixel (DNG SDK algorithm)
    # weight = w[0]*R + w[1]*G + w[2]*B + w[3]*min(RGB) + w[4]*max(RGB)
    r, g, b = img_norm[:, :, 0], img_norm[:, :, 1], img_norm[:, :, 2]
    rgb_min = np.minimum(np.minimum(r, g), b)
    rgb_max = np.maximum(np.maximum(r, g), b)

    weight = (
        input_weights[0] * r
        + input_weights[1] * g
        + input_weights[2] * b
        + input_weights[3] * rgb_min
        + input_weights[4] * rgb_max
    )

    # Scale by baseline exposure and clamp to [0, 1]
    # (DNG SDK dng_reference.cpp:3385 multiplies weight by exposureWeightGain)
    weight = np.clip(weight * baseline_exposure, 0.0, 1.0)

    # Map weight (0-1) to table index (0 to N-1) for interpolation
    # The interpolator axis is 0, 1, ..., N-1, so weight=1 should map to N-1
    weight_idx = weight * (num_table_points - 1)
    weight_idx = np.clip(weight_idx, 0, num_table_points - 1)

    # Create pixel coordinates with half-pixel offset for center sampling
    # (DNG SDK dng_reference.cpp:3326-3327: y = top + 0.5f, x = left + 0.5f)
    rows = np.arange(height, dtype=np.float32) + 0.5
    cols = np.arange(width, dtype=np.float32) + 0.5

    # Normalize to 0-1 image coordinates
    # (DNG SDK dng_reference.cpp:3338-3339)
    v_image = rows / height
    u_image = cols / width

    # Apply origin and spacing transformation to get map coordinates
    # (DNG SDK dng_reference.cpp:3343-3344: x_map = (u_image - origin) / spacing)
    row_coords = (v_image - origin_v) / spacing_v
    col_coords = (u_image - origin_h) / spacing_h

    # Clamp to valid table range
    # (DNG SDK dng_reference.cpp:3348-3349)
    row_coords = np.clip(row_coords, 0, points_v - 1)
    col_coords = np.clip(col_coords, 0, points_h - 1)

    # Create coordinate grids
    row_grid, col_grid = np.meshgrid(row_coords, col_coords, indexing="ij")

    # Build interpolator for 3D table
    v_axis = np.arange(points_v)
    h_axis = np.arange(points_h)
    n_axis = np.arange(num_table_points)
    interp = RegularGridInterpolator(
        (v_axis, h_axis, n_axis),
        table,
        method="linear",
        bounds_error=False,
        fill_value=1.0,  # Gain of 1.0 for out-of-bounds
    )

    # Stack coordinates for interpolation: (height, width) -> (height*width, 3)
    coords = np.stack([row_grid.ravel(), col_grid.ravel(), weight_idx.ravel()], axis=1)

    # Interpolate gains
    gains = interp(coords).reshape(height, width)

    # Apply gain to all channels
    img[:, :, 0] *= gains
    img[:, :, 1] *= gains
    img[:, :, 2] *= gains

    # Clamp and convert back to big-endian 16-bit
    img = np.clip(img, 0, 65535).astype(">u2")

    # Write PPM back
    with open(ppm_path, "wb") as f:
        f.write(magic)
        f.write(f"{width} {height}\n".encode())
        f.write(f"{maxval}\n".encode())
        f.write(img.tobytes())


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
    """Apply tone curve using DNG SDK's max/min interpolation method.

    The DNG SDK algorithm (from dng_reference.cpp RefBaselineRGBTone):
    1. For each pixel, identify max, mid, min RGB values
    2. Apply curve to max and min only
    3. Interpolate mid: mid' = min' + (max' - min') * (mid - min) / (max - min)

    This preserves hue by maintaining the ratio between RGB values,
    which is more accurate than luminance-based scaling.

    Args:
        ppm_path: Path to 16-bit PPM file (modified in place).
        lut: Lookup table from _build_tone_curve_lut().
    """
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

    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    # Allocate output arrays
    rr = np.empty_like(r)
    gg = np.empty_like(g)
    bb = np.empty_like(b)

    # Helper to apply curve via LUT (input 0-65535, output 0-65535)
    def curve(x: np.ndarray) -> np.ndarray:
        return lut[np.clip(x, 0, 65535).astype(np.uint16)].astype(np.float32)

    # DNG SDK algorithm: apply curve to max and min, interpolate mid
    # We need to handle 8 cases based on RGB ordering

    # Find max and min per pixel
    rgb_max = np.maximum(np.maximum(r, g), b)
    rgb_min = np.minimum(np.minimum(r, g), b)

    # Apply curve to max and min
    max_curved = curve(rgb_max)
    min_curved = curve(rgb_min)

    # Compute the interpolation factor for middle value
    # mid' = min' + (max' - min') * (mid - min) / (max - min)
    # Handle division by zero when max == min (grayscale pixels)
    range_orig = rgb_max - rgb_min
    range_curved = max_curved - min_curved

    # For grayscale pixels (max == min), all outputs are the same
    grayscale_mask = range_orig < 1e-6

    # Case 1: r >= g >= b (r is max, b is min, g is mid)
    case1 = (r >= g) & (g >= b) & ~grayscale_mask
    if np.any(case1):
        t = np.zeros_like(r)
        np.divide(g - b, range_orig, out=t, where=case1)
        rr[case1] = max_curved[case1]
        bb[case1] = min_curved[case1]
        gg[case1] = min_curved[case1] + range_curved[case1] * t[case1]

    # Case 2: r >= b > g (r is max, g is min, b is mid)
    case2 = (r >= b) & (b > g) & ~grayscale_mask
    if np.any(case2):
        t = np.zeros_like(r)
        np.divide(b - g, range_orig, out=t, where=case2)
        rr[case2] = max_curved[case2]
        gg[case2] = min_curved[case2]
        bb[case2] = min_curved[case2] + range_curved[case2] * t[case2]

    # Case 3: g > r >= b (g is max, b is min, r is mid)
    case3 = (g > r) & (r >= b) & ~grayscale_mask
    if np.any(case3):
        t = np.zeros_like(r)
        np.divide(r - b, range_orig, out=t, where=case3)
        gg[case3] = max_curved[case3]
        bb[case3] = min_curved[case3]
        rr[case3] = min_curved[case3] + range_curved[case3] * t[case3]

    # Case 4: g >= b > r (g is max, r is min, b is mid)
    case4 = (g >= b) & (b > r) & ~grayscale_mask
    if np.any(case4):
        t = np.zeros_like(r)
        np.divide(b - r, range_orig, out=t, where=case4)
        gg[case4] = max_curved[case4]
        rr[case4] = min_curved[case4]
        bb[case4] = min_curved[case4] + range_curved[case4] * t[case4]

    # Case 5: b > g > r (b is max, r is min, g is mid)
    case5 = (b > g) & (g > r) & ~grayscale_mask
    if np.any(case5):
        t = np.zeros_like(r)
        np.divide(g - r, range_orig, out=t, where=case5)
        bb[case5] = max_curved[case5]
        rr[case5] = min_curved[case5]
        gg[case5] = min_curved[case5] + range_curved[case5] * t[case5]

    # Case 6: b > r >= g (b is max, g is min, r is mid)
    case6 = (b > r) & (r >= g) & ~grayscale_mask
    if np.any(case6):
        t = np.zeros_like(r)
        np.divide(r - g, range_orig, out=t, where=case6)
        bb[case6] = max_curved[case6]
        gg[case6] = min_curved[case6]
        rr[case6] = min_curved[case6] + range_curved[case6] * t[case6]

    # Grayscale case: all channels get the same curved value
    if np.any(grayscale_mask):
        curved_val = curve(r)  # r == g == b for grayscale
        rr[grayscale_mask] = curved_val[grayscale_mask]
        gg[grayscale_mask] = curved_val[grayscale_mask]
        bb[grayscale_mask] = curved_val[grayscale_mask]

    # Store results back
    img[:, :, 0] = rr
    img[:, :, 1] = gg
    img[:, :, 2] = bb

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

    # Calculate luminance (Display P3 coefficients, not BT.709)
    luminance = (
        P3_LUMA_R * img[:, :, 0] + P3_LUMA_G * img[:, :, 1] + P3_LUMA_B * img[:, :, 2]
    )

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


def _apply_display_gamma(ppm_path: Path) -> None:
    """Apply sRGB transfer function for display encoding.

    When dcraw outputs linear data (-g 1 1), we need to apply the sRGB/Display P3
    transfer function before saving to JXL with an embedded Display P3 profile.

    The sRGB transfer function is:
    - Linear region: out = 12.92 * in, for in <= 0.0031308
    - Gamma region: out = 1.055 * in^(1/2.4) - 0.055, for in > 0.0031308

    Args:
        ppm_path: Path to 16-bit linear PPM file (modified in place).
    """
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

    # Normalize to 0-1 range
    img = img / 65535.0

    # Apply sRGB transfer function
    # Linear region: out = 12.92 * in
    # Gamma region: out = 1.055 * in^(1/2.4) - 0.055
    linear_threshold = 0.0031308
    linear_mask = img <= linear_threshold
    img = np.where(
        linear_mask,
        12.92 * img,
        1.055 * np.power(np.maximum(img, 1e-10), 1.0 / 2.4) - 0.055,
    )

    # Convert back to 16-bit
    img = np.clip(img * 65535.0, 0, 65535).astype(">u2")

    # Write PPM back
    with open(ppm_path, "wb") as f:
        f.write(magic)
        f.write(f"{width} {height}\n".encode())
        f.write(f"{maxval}\n".encode())
        f.write(img.tobytes())


def _apply_exposure_ramp(ppm_path: Path, exposure_ev: float) -> None:
    """Apply exposure ramp after PGTM (DNG SDK approach).

    The DNG SDK applies exposure AFTER ProfileGainTableMap, not before.
    This function mimics dng_function_exposure_ramp from dng_render.cpp.

    The exposure ramp is: output = clamp(input * 2^exposure, 0, 1)

    For positive exposure:
    - white_point = 1.0 / 2^exposure
    - Pixels at white_point map to 1.0 (new white)
    - Pixels above white_point clip to 1.0

    Example: BaselineExposure = 1.94 EV
    - exposure_mult = 2^1.94 = 3.84
    - white_point = 1.0 / 3.84 = 0.26
    - Input 0.26 -> Output 1.0

    Args:
        ppm_path: Path to 16-bit linear PPM file (modified in place).
        exposure_ev: Exposure value in EV stops.
    """
    if abs(exposure_ev) < 0.001:
        return  # No exposure adjustment needed

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

    # Normalize to 0-1 range
    img = img / 65535.0

    # Apply exposure as linear multiplier (DNG SDK's exposure ramp)
    exposure_mult = 2.0**exposure_ev
    img = img * exposure_mult

    # Clip to valid range (hard clipping like DNG SDK)
    img = np.clip(img, 0.0, 1.0)

    # Convert back to 16-bit
    img = (img * 65535.0).astype(">u2")

    # Write PPM back
    with open(ppm_path, "wb") as f:
        f.write(magic)
        f.write(f"{width} {height}\n".encode())
        f.write(f"{maxval}\n".encode())
        f.write(img.tobytes())


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
    - Output to 16-bit PPM in DCI-P3 color space

    Phase 2 (apple style only - PGTM local tone mapping):
    - Extract ProfileGainTableMap from DNG (DNG 1.6 local tone mapping)
    - Apply spatially-varying gain to lift shadows, control highlights
    - This creates the characteristic "Apple look"

    Phase 3 (global tone curve):
    - Apply tone curve (apple=ProfileToneCurve from DNG, vivid/film=preset)
    - Flat style uses linear curve, linear style skips entirely

    Phase 4 (saturation adjustment):
    - Apply saturation adjustment per style

    Styles:
    - apple: Match Apple Photos (PGTM + ProfileToneCurve + slight saturation boost)
    - flat: Low contrast for grading (linear curve, -W flag)
    - vivid: High contrast, punchy (aggressive S-curve, +10% saturation)
    - film: Soft contrast, warm (gentle S-curve, -5% saturation)
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
        # For flat: apply only BaselineExposure (no arbitrary boost), use sRGB gamma
        base_exp = _get_baseline_exposure(info.path)
        exp_shift = base_exp  # Use per-image BaselineExposure, no additional boost
        highlight_preserve = 0.0
        highlight_mode = "0"  # Clip highlights
        use_srgb_gamma = True
    elif style == "linear":
        # For linear: raw sensor data, no adjustments
        exp_shift = 1.0
        highlight_preserve = 0.0
        highlight_mode = "0"  # Clip highlights
        use_srgb_gamma = False
    elif style == "apple":
        # For apple: no exposure in dcraw - we apply it in Python after PGTM
        # (following the DNG SDK pipeline order)
        baseline_exposure_ev = _get_baseline_exposure_ev(info.path)
        exp_shift = 1.0  # No exposure adjustment in dcraw
        highlight_preserve = 0.0
        highlight_mode = "0"  # Clip highlights (exposure applied in Python)
        use_srgb_gamma = False
    else:
        # For vivid/film: use per-image BaselineExposure in dcraw
        exp_shift = _get_baseline_exposure(info.path)
        highlight_preserve = 0.5
        highlight_mode = "2"  # Blend highlights
        use_srgb_gamma = False
        baseline_exposure_ev = None  # Not used

    try:
        # Build environment with library path for dcraw_emu_dng
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = _get_dcraw_library_path()

        # Phase 1: Process RAW to 16-bit PPM using dcraw_emu_dng with DNG SDK
        dcraw_cmd = [
            str(dcraw_emu),
            "-dngsdk",  # Use DNG SDK to decode JXL-compressed RAW
            "-6",  # 16-bit output
            "-q",
            "3",  # AHD demosaicing (high quality)
            "-H",
            highlight_mode,  # Highlight handling
            "-o",
            "6",  # Output color space: DCI-P3 D65 (verified from LibRaw source)
            "-Z",
            str(temp_ppm),
            str(info.path),
        ]

        # Add exposure adjustment for styles that apply it in dcraw (not apple)
        if style != "apple":
            dcraw_cmd.insert(7, "-aexpo")
            dcraw_cmd.insert(8, f"{exp_shift:.4f}")
            dcraw_cmd.insert(9, f"{highlight_preserve:.1f}")

        # Add -W flag to disable auto-brightness (needed for all styles except default)
        if style in ("flat", "linear", "apple", "vivid", "film"):
            dcraw_cmd.insert(1, "-W")

        # Add gamma flags based on style
        if use_srgb_gamma:
            # sRGB gamma for flat style (2.4 gamma with 12.92 linear slope)
            dcraw_cmd.insert(1, "-g")
            dcraw_cmd.insert(2, "2.4")
            dcraw_cmd.insert(3, "12.92")
        elif style in ("apple", "vivid", "film"):
            # Linear output for styles that apply PGTM/tone curve in Python
            # (we apply display gamma after all processing)
            dcraw_cmd.insert(1, "-g")
            dcraw_cmd.insert(2, "1")
            dcraw_cmd.insert(3, "1")

        result = subprocess.run(dcraw_cmd, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            raise JxlExtractionError(
                f"dcraw_emu_dng failed: {result.stderr or result.stdout}"
            )

        if not temp_ppm.exists():
            raise JxlExtractionError(
                f"dcraw_emu_dng did not produce output file for {info.path.name}"
            )

        # Phase 2: Apply ProfileGainTableMap for apple style (local tone mapping)
        # Note: PGTM uses exp_weight = 2^BaselineExposure for weight scaling
        # (compensation for applying PGTM before exposure)
        if style == "apple":
            pgtm = _extract_profile_gain_table_map(info.path)
            pgtm_exp_mult = 2.0**baseline_exposure_ev  # PGTM weight scaling
            if pgtm:
                _apply_profile_gain_table_map(temp_ppm, pgtm, pgtm_exp_mult)

        # Phase 2.5: Apply exposure ramp AFTER PGTM (DNG SDK pipeline order)
        # This is the actual exposure boost - applied as simple linear multiply
        if style == "apple":
            _apply_exposure_ramp(temp_ppm, baseline_exposure_ev)

        # Phase 3: Apply global tone curve based on style
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

        # Phase 4: Apply saturation adjustment
        sat_factor = SATURATION_ADJUSTMENTS.get(style, 1.0)
        if sat_factor != 1.0:
            _adjust_saturation(temp_ppm, sat_factor)

        # Phase 5: Apply display gamma for styles using linear dcraw output
        # (sRGB transfer function for Display P3 encoding)
        if style in ("apple", "vivid", "film"):
            _apply_display_gamma(temp_ppm)

        # Encode 16-bit PPM to lossless JXL with Display P3 color profile
        cjxl_cmd = [
            "cjxl",
            str(temp_ppm),
            str(output_path),
            "-d",
            "0",  # Lossless (distance 0)
            "-e",
            "7",  # Encoding effort (1-10)
            "-x",
            "color_space=DisplayP3",  # Embed Display P3 color profile
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


def process_all(
    images: list[tuple[ImageInfo, Path]],
    max_workers: int,
) -> None:
    """Process all images with parallel execution.

    Raises:
        JxlExtractionError: On image extraction failure.
    """
    dest_dir = Path(DESTINATION_DIR)

    total = len(images)
    console.print(f"\n[bold blue]Processing {total} image(s)[/]")
    console.print(f"  Style: {TONE_CURVE_STYLE}")
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
    console.print("\n[bold]Photo Converter[/] (DNG to JXL + Image Copy)\n")

    try:
        # Validate environment
        validate_environment()

        source_dir = Path(SOURCE_DIR)
        dest_dir = Path(DESTINATION_DIR)

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
        process_all(images_to_process, MAX_WORKERS)

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
