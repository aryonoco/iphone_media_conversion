#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy>=2.0",
#     "rich>=13.0.0",
# ]
# ///
#
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2025-2026 Aryan Ameri
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
Add ISO 21496-1 HDR Gain Map to Lightroom-exported JXL files.

Takes a Lightroom-exported JXL (Rec.2020/BT.709 SDR) and the original ProRAW DNG,
produces a new JXL with gain map for proper HDR display.

The key insight: preserve Lightroom's beautiful rendering as the SDR base and
only extract HDR luminance headroom from the DNG to create the gain map.
"""

from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from jxl_container import JxlBox, extract_naked_codestream, insert_jhgm_box, parse_jxl_container, write_jxl_container

__all__: Final[list[str]] = [
    "create_gain_map_metadata",
    "create_jhgm_box",
    "align_hdr_to_sdr",
    "find_min_max_without_outliers",
    "compute_gain_map",
    "process_jxl_with_gainmap",
    "main",
]

__version__: Final[str] = "1.0.0"

# Paths to external tools
SCRIPT_DIR: Final[Path] = Path(__file__).resolve().parent
DJXL_PATH: Final[Path] = SCRIPT_DIR / "build" / "deps" / "bin" / "djxl"
CJXL_PATH: Final[Path] = SCRIPT_DIR / "build" / "deps" / "bin" / "cjxl"
DCRAW_EMU_PATH: Final[Path] = SCRIPT_DIR / "build" / "dcraw_emu_dng"

# Library paths for external tools
LIB_PATHS: Final[list[Path]] = [
    SCRIPT_DIR / "build" / "deps" / "lib",
    SCRIPT_DIR / "build" / "deps" / "lib64",
    SCRIPT_DIR / "build" / "lib",
]

# Rec.2020 luminance coefficients
Y_COEFFS: Final[NDArray[np.float64]] = np.array([0.2627, 0.6780, 0.0593])

# Console for rich output
console = Console()


# =============================================================================
# Environment Setup
# =============================================================================


def get_library_path() -> str:
    """Get LD_LIBRARY_PATH for external tools."""
    paths = [str(p) for p in LIB_PATHS if p.exists()]
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    if existing:
        paths.append(existing)
    return ":".join(paths)


def run_command(
    cmd: list[str],
    *,
    check: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a command with proper library paths."""
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = get_library_path()

    return subprocess.run(
        cmd,
        env=env,
        check=check,
        capture_output=capture_output,
        text=True,
    )


# =============================================================================
# ISO 21496-1 Metadata Encoder
# =============================================================================


def float_to_fraction(value: float, max_denominator: int = 134217728) -> tuple[int, int]:
    """Convert float to fraction with reasonable precision.

    Args:
        value: Float value to convert
        max_denominator: Maximum denominator value

    Returns:
        Tuple of (numerator, denominator)
    """
    frac = Fraction(value).limit_denominator(max_denominator)
    return frac.numerator, frac.denominator


def create_gain_map_metadata(
    gain_min: float,
    gain_max: float,
    gamma: float,
    base_headroom: float,
    alt_headroom: float,
    base_offset: float = 1 / 64,
    alt_offset: float = 1 / 64,
    single_channel: bool = True,
) -> bytes:
    """Serialize ISO 21496-1 gain map metadata (big-endian).

    Binary format (verified from AVIF files created by avifgainmaputil):
    - minimum_version: uint16 BE (= 0)
    - writer_version: uint16 BE (= 0)
    - flags: uint8 (bit7=is_multichannel, bit6=use_base_color_space)
    - base_headroom: uint32 BE numerator + uint32 BE denominator
    - alternate_headroom: uint32 BE numerator + uint32 BE denominator
    - Per channel (1 or 3 times):
      - gain_min: int32 BE (signed) + uint32 BE denominator
      - gain_max: int32 BE (signed) + uint32 BE denominator
      - gamma: uint32 BE + uint32 BE
      - base_offset: int32 BE (signed) + uint32 BE
      - alternate_offset: int32 BE (signed) + uint32 BE

    Args:
        gain_min: Minimum gain value (log2 scale)
        gain_max: Maximum gain value (log2 scale)
        gamma: Gamma value for gain map encoding
        base_headroom: Base (SDR) headroom in log2 scale
        alt_headroom: Alternate (HDR) headroom in log2 scale
        base_offset: Base offset for numerical stability
        alt_offset: Alternate offset for numerical stability
        single_channel: If True, use single-channel (luminance only)

    Returns:
        Serialized metadata bytes
    """
    parts: list[bytes] = []

    # Header
    parts.append(struct.pack(">H", 0))  # minimum_version
    parts.append(struct.pack(">H", 0))  # writer_version

    # Flags: bit7=is_multichannel, bit6=use_base_color_space, bits5-0=reserved
    # use_base_color_space=0 (false): use alternate color space for gain map math
    # Since both SDR and HDR use Rec.2020, this flag is irrelevant for our use case
    is_multichannel = 0 if single_channel else 1
    use_base_color_space = 0  # Explicit: false, matches avifgainmaputil output
    flags = (is_multichannel << 7) | (use_base_color_space << 6)
    parts.append(struct.pack("B", flags))

    # Headrooms (unsigned fractions)
    n, d = float_to_fraction(base_headroom)
    parts.append(struct.pack(">II", max(0, n), d))  # Clamp numerator to unsigned
    n, d = float_to_fraction(alt_headroom)
    parts.append(struct.pack(">II", max(0, n), d))

    # Per-channel data (1 or 3 times)
    channel_count = 1 if single_channel else 3
    for _ in range(channel_count):
        # gain_min (signed numerator)
        n, d = float_to_fraction(gain_min)
        parts.append(struct.pack(">iI", n, d))

        # gain_max (signed numerator)
        n, d = float_to_fraction(gain_max)
        parts.append(struct.pack(">iI", n, d))

        # gamma (unsigned)
        n, d = float_to_fraction(gamma)
        parts.append(struct.pack(">II", n, d))

        # base_offset (signed numerator)
        n, d = float_to_fraction(base_offset)
        parts.append(struct.pack(">iI", n, d))

        # alternate_offset (signed numerator)
        n, d = float_to_fraction(alt_offset)
        parts.append(struct.pack(">iI", n, d))

    return b"".join(parts)


# =============================================================================
# JHGM Box Creator
# =============================================================================


def create_jhgm_box(
    metadata: bytes,
    gain_map_codestream: bytes,
    color_encoding: bytes | None = None,
    alt_icc: bytes | None = None,
) -> bytes:
    """Create jhgm box contents per libjxl gain_map.h.

    Format:
    - jhgm_version: uint8 (0)
    - metadata_size: uint16 BE
    - metadata: bytes (ISO 21496-1 binary)
    - color_encoding_size: uint8 (0 if none)
    - color_encoding: bytes (optional)
    - alt_icc_size: uint32 BE (0 if none)
    - alt_icc: bytes (optional, compressed)
    - gain_map: remaining bytes (JXL naked codestream)

    Args:
        metadata: ISO 21496-1 metadata bytes
        gain_map_codestream: JXL naked codestream for gain map
        color_encoding: Optional JXL ColorEncoding bytes
        alt_icc: Optional compressed ICC profile bytes

    Returns:
        Complete jhgm box data
    """
    parts: list[bytes] = []

    # jhgm_version
    parts.append(struct.pack("B", 0))

    # metadata_size and metadata
    parts.append(struct.pack(">H", len(metadata)))
    parts.append(metadata)

    # color_encoding_size and color_encoding
    if color_encoding:
        # Size is in bits, rounded up to bytes
        size_bits = len(color_encoding) * 8
        parts.append(struct.pack("B", size_bits))
        parts.append(color_encoding)
    else:
        parts.append(struct.pack("B", 0))

    # alt_icc_size and alt_icc
    if alt_icc:
        parts.append(struct.pack(">I", len(alt_icc)))
        parts.append(alt_icc)
    else:
        parts.append(struct.pack(">I", 0))

    # gain_map codestream
    parts.append(gain_map_codestream)

    return b"".join(parts)


# =============================================================================
# Mid-Tone Alignment
# =============================================================================


def align_hdr_to_sdr(
    sdr_linear: NDArray[np.float32],
    hdr_linear: NDArray[np.float32],
    low_percentile: float = 20.0,
    high_percentile: float = 80.0,
) -> NDArray[np.float32]:
    """Scale HDR so its mid-tone luminance matches SDR's mid-tones.

    Uses percentile band (20-80%) for robust alignment across scene types.

    Why percentile band instead of median:
    - Uses more data points (all values in range, not just one)
    - Explicitly excludes extreme shadows and highlights
    - More stable for unusual distributions (dark scenes, bimodal, etc.)

    Args:
        sdr_linear: SDR image in linear Rec.2020 (H, W, 3)
        hdr_linear: HDR image in linear Rec.2020 (H, W, 3)
        low_percentile: Lower bound of mid-tone range
        high_percentile: Upper bound of mid-tone range

    Returns:
        Aligned HDR array (same shape as input)
    """
    sdr_lum = np.dot(sdr_linear, Y_COEFFS).flatten()
    hdr_lum = np.dot(hdr_linear, Y_COEFFS).flatten()

    # Compute percentile bounds from SDR (the reference)
    sdr_low = np.percentile(sdr_lum, low_percentile)
    sdr_high = np.percentile(sdr_lum, high_percentile)

    # Create mask for mid-tone pixels in SDR
    sdr_mask = (sdr_lum >= sdr_low) & (sdr_lum <= sdr_high)

    # Use corresponding HDR pixels (same spatial locations)
    sdr_midtones = sdr_lum[sdr_mask]
    hdr_midtones = hdr_lum[sdr_mask]

    # Compute mean of mid-tone bands
    sdr_mean = float(np.mean(sdr_midtones))
    hdr_mean = float(np.mean(hdr_midtones))

    if hdr_mean > 1e-10:
        scale = sdr_mean / hdr_mean
        return (hdr_linear * scale).astype(np.float32)
    return hdr_linear


# =============================================================================
# Outlier Rejection
# =============================================================================


def find_min_max_without_outliers(
    gain_values: NDArray[np.float32],
    outlier_ratio: float = 0.001,
    bucket_size: float = 0.01,
    max_buckets: int = 10000,
) -> tuple[float, float]:
    """Find min/max of gain values, discarding outliers.

    Algorithm (from libavif gainmap.c lines 358-413):
    1. Find absolute min/max
    2. Create histogram with bucket_size granularity
    3. Walk from each end, skipping empty buckets until
       we've passed more than (0.1%/2) of total pixels
    4. The last empty bucket boundary becomes the new min/max

    Args:
        gain_values: 1D array of log2 gain values
        outlier_ratio: Ratio of outliers to discard (default 0.1%)
        bucket_size: Histogram bucket size
        max_buckets: Maximum number of histogram buckets

    Returns:
        Tuple of (range_min, range_max)
    """
    if len(gain_values) == 0:
        return 0.0, 0.0

    # Step 1: Find absolute range
    abs_min = float(np.min(gain_values))
    abs_max = float(np.max(gain_values))

    range_span = abs_max - abs_min
    if range_span <= bucket_size * 2:
        return abs_min, abs_max

    # Step 2: Create histogram
    num_buckets = min(int(np.ceil(range_span / bucket_size)), max_buckets)
    histogram, bin_edges = np.histogram(
        gain_values, bins=num_buckets, range=(abs_min, abs_max)
    )

    max_outliers_each_side = int(round(len(gain_values) * outlier_ratio / 2.0))
    if max_outliers_each_side == 0:
        return abs_min, abs_max

    # Step 3: Walk from left, find new min
    range_min = abs_min
    left_outliers = 0
    for i in range(num_buckets):
        left_outliers += histogram[i]
        if left_outliers > max_outliers_each_side:
            break
        if histogram[i] == 0:
            # Use upper edge of this empty bucket
            range_min = float(bin_edges[i + 1])

    # Step 4: Walk from right, find new max
    range_max = abs_max
    right_outliers = 0
    for i in range(num_buckets - 1, -1, -1):
        right_outliers += histogram[i]
        if right_outliers > max_outliers_each_side:
            break
        if histogram[i] == 0:
            # Use lower edge of this empty bucket
            range_max = float(bin_edges[i])

    return range_min, range_max


# =============================================================================
# Gain Map Computation
# =============================================================================


def compute_gain_map(
    sdr_linear: NDArray[np.float32],
    hdr_aligned: NDArray[np.float32],
    offset: float = 1 / 64,
    epsilon: float = 1e-10,
) -> tuple[NDArray[np.float32], float, float, float, float]:
    """Compute luminance-based gain map.

    Args:
        sdr_linear: SDR image in linear Rec.2020 (H, W, 3)
        hdr_aligned: Aligned HDR image in linear Rec.2020 (H, W, 3)
        offset: Offset for numerical stability (ISO 21496-1 default: 1/64)
        epsilon: Small value to prevent log2(0)

    Returns:
        Tuple of:
        - gain_normalized: (H, W) array in [0, 1]
        - gain_min: log2 minimum
        - gain_max: log2 maximum
        - base_headroom: log2(max SDR luminance)
        - alt_headroom: log2(max HDR luminance)
    """
    sdr_lum = np.dot(sdr_linear, Y_COEFFS)
    hdr_lum = np.dot(hdr_aligned, Y_COEFFS)

    # Clamp luminance to non-negative before ratio computation
    # Negative values can occur from color transforms or decoder quirks
    sdr_lum = np.maximum(sdr_lum, 0.0)
    hdr_lum = np.maximum(hdr_lum, 0.0)

    # Compute ratio with offset, clamp to epsilon before log2
    # Per libavif: AVIF_MAX(ratio, kEpsilon) before log2f()
    ratio = (hdr_lum + offset) / (sdr_lum + offset)
    ratio = np.maximum(ratio, epsilon)
    gain_log2 = np.log2(ratio)

    # Find min/max discarding outliers (per libavif's avifFindMinMaxWithoutOutliers)
    gain_min, gain_max = find_min_max_without_outliers(gain_log2.flatten().astype(np.float32))

    # Normalize to [0, 1]
    if gain_max > gain_min:
        gain_normalized = (gain_log2 - gain_min) / (gain_max - gain_min)
    else:
        gain_normalized = np.zeros_like(gain_log2)

    gain_normalized = np.clip(gain_normalized, 0, 1).astype(np.float32)

    # Compute headrooms per libavif: log2(AVIF_MAX(max_pixel_value, kEpsilon))
    # libavif initializes to 1.0f and tracks actual max during iteration
    base_max = max(float(sdr_lum.max()), 1.0, epsilon)
    alt_max = max(float(hdr_lum.max()), 1.0, epsilon)
    base_headroom = float(np.log2(base_max))
    alt_headroom = float(np.log2(alt_max))

    return gain_normalized, gain_min, gain_max, base_headroom, alt_headroom


# =============================================================================
# External Tool Wrappers
# =============================================================================


def decode_jxl_to_linear(jxl_path: Path, output_ppm: Path) -> None:
    """Decode JXL to linear Rec.2020 PPM using djxl.

    SCALING ASSUMPTION: djxl outputs normalized values where:
    - SDR peak white = 1.0
    - Values are in linear light (no gamma)
    - Color space is Rec.2020 primaries
    """
    cmd = [
        str(DJXL_PATH),
        str(jxl_path),
        str(output_ppm),
        "--color_space=RGB_D65_202_Rel_Lin",
    ]
    result = run_command(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"djxl failed: {result.stderr}")


def decode_dng_to_linear(dng_path: Path, output_ppm: Path) -> None:
    """Decode DNG to linear Rec.2020 PPM with HDR highlights preserved."""
    cmd = [
        str(DCRAW_EMU_PATH),
        "-4",  # 16-bit linear (equivalent to -6 -W -g 1 1)
        "-H", "1",  # Unclip highlights (preserve HDR)
        "-o", "8",  # Rec.2020 output
        "-q", "3",  # AHD demosaicing
        "-dngsdk",  # Use DNG SDK for JXL-compressed DNG
        "-Z", str(output_ppm),
        str(dng_path),
    ]
    result = run_command(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"dcraw_emu_dng failed: {result.stderr}")


def encode_gain_map_jxl(pgm_path: Path, output_jxl: Path) -> None:
    """Encode grayscale gain map as JXL naked codestream."""
    cmd = [
        str(CJXL_PATH),
        str(pgm_path),
        str(output_jxl),
        "-d", "1.0",  # Visually lossless
        "-e", "7",  # High effort
        "--container=0",  # Naked codestream (no container)
    ]
    result = run_command(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"cjxl failed: {result.stderr}")


# =============================================================================
# PPM/PGM I/O
# =============================================================================


def read_ppm_as_float(path: Path) -> NDArray[np.float32]:
    """Read PPM file as float32 array normalized to [0, 1+]."""
    with open(path, "rb") as f:
        # Read header
        magic = f.readline().decode("ascii").strip()
        if magic not in ("P5", "P6"):
            raise ValueError(f"Unsupported PNM format: {magic}")

        # Skip comments
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()

        # Read dimensions
        parts = line.decode("ascii").split()
        width, height = int(parts[0]), int(parts[1])

        # Read maxval
        maxval = int(f.readline().decode("ascii").strip())

        # Read pixel data
        data = f.read()

    if maxval == 65535:
        # 16-bit big-endian
        pixels = np.frombuffer(data, dtype=">u2")
    elif maxval == 255:
        # 8-bit
        pixels = np.frombuffer(data, dtype=np.uint8)
    else:
        raise ValueError(f"Unsupported maxval: {maxval}")

    # Reshape based on format
    if magic == "P6":
        pixels = pixels.reshape(height, width, 3)
    else:  # P5
        pixels = pixels.reshape(height, width)

    return (pixels.astype(np.float32) / maxval).astype(np.float32)


def write_grayscale_pgm(path: Path, data: NDArray[np.float32]) -> None:
    """Write grayscale data as PGM (P5 binary format).

    Args:
        path: Output file path
        data: 2D array normalized to [0, 1]
    """
    height, width = data.shape
    # Convert to 16-bit for better precision
    data_16bit = (np.clip(data, 0, 1) * 65535).astype(np.uint16)
    with open(path, "wb") as f:
        f.write(f"P5\n{width} {height}\n65535\n".encode("ascii"))
        # PGM uses big-endian for 16-bit values
        f.write(data_16bit.astype(">u2").tobytes())


# =============================================================================
# Metadata Extraction
# =============================================================================


def get_baseline_exposure(dng_path: Path) -> float:
    """Get BaselineExposure from DNG file using exiftool."""
    try:
        result = subprocess.run(
            ["exiftool", "-BaselineExposure", "-n", "-s", "-s", "-s", str(dng_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return 0.0


def get_orientation(path: Path) -> int:
    """Get EXIF orientation from file using exiftool."""
    try:
        result = subprocess.run(
            ["exiftool", "-Orientation", "-n", "-s", "-s", "-s", str(path)],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return 1


def apply_rotation_to_match(
    hdr_linear: NDArray[np.float32],
    sdr_shape: tuple[int, int],
    dng_path: Path,
) -> NDArray[np.float32]:
    """Apply rotation to HDR to match SDR dimensions.

    Lightroom exports JXL with orientation already applied in the pixels.
    dcraw_emu outputs raw sensor dimensions without applying EXIF rotation.
    We use the DNG's EXIF orientation to determine how to rotate the HDR.

    Args:
        hdr_linear: HDR image from DNG (raw sensor orientation)
        sdr_shape: Target shape (height, width) from SDR/JXL
        dng_path: Path to DNG for reading EXIF orientation

    Returns:
        Rotated HDR array matching SDR dimensions
    """
    hdr_h, hdr_w = hdr_linear.shape[:2]
    sdr_h, sdr_w = sdr_shape

    # If dimensions already match, no rotation needed
    if (hdr_h, hdr_w) == (sdr_h, sdr_w):
        return hdr_linear

    # If dimensions are swapped, we need to rotate
    if (hdr_h, hdr_w) == (sdr_w, sdr_h):
        # Get DNG orientation to determine rotation direction
        dng_orient = get_orientation(dng_path)

        # EXIF orientation values:
        # 1 = Normal (no rotation needed)
        # 3 = Rotated 180°
        # 6 = Rotated 90° CW (camera held portrait, home button right)
        # 8 = Rotated 90° CCW (camera held portrait, home button left)

        if dng_orient == 6:
            # Rotate 90° CW: k=-1 in np.rot90 (or k=3)
            return np.rot90(hdr_linear, k=-1)
        elif dng_orient == 8:
            # Rotate 90° CCW: k=1 in np.rot90
            return np.rot90(hdr_linear, k=1)
        else:
            # Default to 90° CW for unknown orientations where dims are swapped
            return np.rot90(hdr_linear, k=-1)

    # Dimensions don't match and aren't swapped - this is an error
    raise ValueError(
        f"Cannot match dimensions: HDR ({hdr_h}, {hdr_w}) to SDR ({sdr_h}, {sdr_w})"
    )


# =============================================================================
# Main Pipeline
# =============================================================================


def process_jxl_with_gainmap(
    lightroom_jxl: Path,
    dng_path: Path,
    output_jxl: Path,
    *,
    verbose: bool = False,
) -> None:
    """Main processing pipeline.

    Takes a Lightroom-exported JXL and original DNG, produces a new JXL
    with ISO 21496-1 gain map for HDR display.

    Args:
        lightroom_jxl: Path to Lightroom-exported JXL (SDR base)
        dng_path: Path to original ProRAW DNG
        output_jxl: Path for output JXL with gain map
        verbose: Print detailed progress
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # 1. Decode sources to linear Rec.2020
        if verbose:
            console.print("[dim]Decoding JXL to linear Rec.2020...[/dim]")
        sdr_ppm = tmp / "sdr.ppm"
        decode_jxl_to_linear(lightroom_jxl, sdr_ppm)

        if verbose:
            console.print("[dim]Decoding DNG to linear Rec.2020...[/dim]")
        hdr_ppm = tmp / "hdr.ppm"
        decode_dng_to_linear(dng_path, hdr_ppm)

        # 2. Read pixel data
        if verbose:
            console.print("[dim]Reading pixel data...[/dim]")
        sdr_linear = read_ppm_as_float(sdr_ppm)
        hdr_linear = read_ppm_as_float(hdr_ppm)

        # 3. Apply BaselineExposure to HDR
        baseline_ev = get_baseline_exposure(dng_path)
        if verbose:
            console.print(f"[dim]BaselineExposure: {baseline_ev:.2f} EV[/dim]")
        hdr_linear *= 2.0 ** baseline_ev

        # 4. Handle rotation (DNG needs rotation to match JXL which already has rotation applied)
        if verbose:
            console.print(f"[dim]SDR shape: {sdr_linear.shape[:2]}, HDR shape: {hdr_linear.shape[:2]}[/dim]")
        hdr_linear = apply_rotation_to_match(hdr_linear, sdr_linear.shape[:2], dng_path)
        if verbose:
            console.print(f"[dim]After rotation: HDR shape: {hdr_linear.shape[:2]}[/dim]")

        # 5. CRITICAL: Align HDR mid-tones to SDR mid-tones
        if verbose:
            console.print("[dim]Aligning HDR to SDR mid-tones...[/dim]")
        hdr_aligned = align_hdr_to_sdr(sdr_linear, hdr_linear)

        # 6. Compute gain map
        if verbose:
            console.print("[dim]Computing gain map...[/dim]")
        gain_map, gain_min, gain_max, base_hr, alt_hr = compute_gain_map(
            sdr_linear, hdr_aligned
        )

        if verbose:
            console.print(f"[dim]  Gain range: [{gain_min:.3f}, {gain_max:.3f}][/dim]")
            console.print(f"[dim]  Base headroom: {base_hr:.3f} ({2**base_hr:.2f}x)[/dim]")
            console.print(f"[dim]  Alt headroom: {alt_hr:.3f} ({2**alt_hr:.2f}x)[/dim]")

        # 7. Encode gain map as JXL (using PGM for grayscale)
        if verbose:
            console.print("[dim]Encoding gain map as JXL...[/dim]")
        gain_pgm = tmp / "gain.pgm"
        write_grayscale_pgm(gain_pgm, gain_map)
        gain_jxl = tmp / "gain.jxl"
        encode_gain_map_jxl(gain_pgm, gain_jxl)
        gain_codestream = extract_naked_codestream(gain_jxl)

        # 8. Create ISO 21496-1 metadata
        metadata = create_gain_map_metadata(
            gain_min=gain_min,
            gain_max=gain_max,
            gamma=1.0,
            base_headroom=base_hr,
            alt_headroom=alt_hr,
        )

        # 9. Create jhgm box
        jhgm_data = create_jhgm_box(metadata, gain_codestream)
        jhgm_box = JxlBox(box_type="jhgm", data=jhgm_data)

        # 10. Parse original JXL and insert jhgm
        if verbose:
            console.print("[dim]Inserting jhgm box into JXL container...[/dim]")
        boxes = parse_jxl_container(lightroom_jxl)
        boxes = insert_jhgm_box(boxes, jhgm_box)

        # 11. Write output
        write_jxl_container(output_jxl, boxes)

        if verbose:
            console.print(f"[green]Created: {output_jxl}[/green]")


# =============================================================================
# CLI
# =============================================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class FileSet:
    """A matched set of JXL and DNG files."""
    jxl: Path
    dng: Path
    output: Path


def find_file_sets(directory: Path, output_dir: Path | None = None) -> list[FileSet]:
    """Find matching JXL and DNG pairs by filename stem."""
    jxl_files = {f.stem: f for f in directory.glob("*.jxl")}
    dng_files = {f.stem.upper(): f for f in directory.glob("*.DNG")}
    dng_files.update({f.stem.upper(): f for f in directory.glob("*.dng")})

    sets: list[FileSet] = []
    for stem, jxl_path in jxl_files.items():
        dng_path = dng_files.get(stem.upper())
        if dng_path:
            out_dir = output_dir or directory
            output_path = out_dir / f"{stem}_hdr.jxl"
            sets.append(FileSet(jxl=jxl_path, dng=dng_path, output=output_path))

    return sets


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Add ISO 21496-1 HDR gain map to Lightroom-exported JXL files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --source photo.jxl --dng photo.DNG --output photo_hdr.jxl
  %(prog)s --dir /path/to/photos
  %(prog)s --dir /path/to/photos --output-dir /path/to/output

The script preserves Lightroom's SDR rendering and adds HDR highlight
extension from the original DNG file.
""",
    )

    parser.add_argument(
        "--source", "-s",
        type=Path,
        help="Source JXL file (Lightroom export)",
    )
    parser.add_argument(
        "--dng", "-d",
        type=Path,
        help="Original DNG file",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output JXL file with gain map",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        help="Directory to process (finds matching JXL/DNG pairs)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for batch processing",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed progress",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.dir:
        # Batch mode
        if not args.dir.is_dir():
            console.print(f"[red]Error: {args.dir} is not a directory[/red]")
            return 1

        file_sets = find_file_sets(args.dir, args.output_dir)
        if not file_sets:
            console.print("[yellow]No matching JXL/DNG pairs found[/yellow]")
            return 0

        console.print(f"Found {len(file_sets)} file pairs to process")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing", total=len(file_sets))

            for fs in file_sets:
                progress.update(task, description=f"Processing {fs.jxl.name}")
                try:
                    process_jxl_with_gainmap(
                        fs.jxl, fs.dng, fs.output, verbose=args.verbose
                    )
                except Exception as e:
                    console.print(f"[red]Error processing {fs.jxl.name}: {e}[/red]")
                progress.advance(task)

    elif args.source and args.dng:
        # Single file mode
        if not args.source.exists():
            console.print(f"[red]Error: {args.source} not found[/red]")
            return 1
        if not args.dng.exists():
            console.print(f"[red]Error: {args.dng} not found[/red]")
            return 1

        output = args.output or args.source.with_stem(args.source.stem + "_hdr")

        try:
            process_jxl_with_gainmap(args.source, args.dng, output, verbose=args.verbose)
            console.print(f"[green]Created: {output}[/green]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return 1
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
