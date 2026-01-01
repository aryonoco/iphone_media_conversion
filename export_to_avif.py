#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "rich>=14.0",
#     "numpy>=1.26",
#     "pypng==0.20220715.0",
# ]
# ///
#
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2025-2026 Aryan Ameri
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#

"""
AVIF Gain Map Exporter

Converts iPhone 16/17 Pro ProRAW DNG files to ISO 21496-1 gain map AVIF
with SDR base layer + HDR alternate layer.

Output format:
    - SDR base: Rec.2020 + sRGB transfer (CICP 9/13/0)
    - HDR alternate: Rec.2020 + PQ transfer (CICP 9/16/0)
    - Gain map: Encodes log2(HDR/SDR) luminance ratio

Prerequisites:
    - Run libraw_dng.py to build dcraw_emu_dng
    - Run avif_tools.py to build avifenc and avifgainmaputil
    - Requires: exiftool

Usage:
    ./export_to_avif.py --source ~/Photos/raw --dest ~/Photos/avif
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Self, TypedDict

import numpy as np
from numpy.typing import NDArray
import png

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
DESTINATION_DIR: str = "/var/home/admin/Pictures/avif-output"

# ═══════════════════════════════════════════════════════════════════
#                        END CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

# Tool paths (relative to script directory)
SCRIPT_DIR: Final[Path] = Path(__file__).resolve().parent
AVIFENC: Final[Path] = SCRIPT_DIR / "avif_build" / "bin" / "avifenc"
AVIFGAINMAPUTIL: Final[Path] = SCRIPT_DIR / "avif_build" / "bin" / "avifgainmaputil"
DCRAW_EMU: Final[Path] = SCRIPT_DIR / "build" / "dcraw_emu_dng"

# Rich console for output
console = Console()

# Global flags (set by argument parser)
DEBUG_MODE: bool = False
DEBUG_OUTPUT_DIR: Path | None = None


# ═══════════════════════════════════════════════════════════════════
#                        TYPE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════


class PGTMData(TypedDict):
    """ProfileGainTableMap data structure from DNG 1.6."""

    points_v: int
    points_h: int
    spacing_v: float
    spacing_h: float
    origin_v: float
    origin_h: float
    num_table_points: int
    input_weights: tuple[float, float, float, float, float]  # R, G, B, min, max
    table: NDArray[np.float32]  # Shape: (V, H, N)


# ═══════════════════════════════════════════════════════════════════
#                        EXCEPTIONS
# ═══════════════════════════════════════════════════════════════════


class AvifExportError(Exception):
    """Base exception for AVIF export errors."""


class ValidationError(AvifExportError):
    """Raised when environment validation fails."""


class DngProcessingError(AvifExportError):
    """Raised when DNG processing fails."""


class AvifEncodingError(AvifExportError):
    """Raised when AVIF encoding fails."""


class MetadataError(AvifExportError):
    """Raised when metadata transfer fails."""


# ═══════════════════════════════════════════════════════════════════
#                        DATA MODELS
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageInfo:
    """Immutable DNG file metadata from exiftool."""

    path: Path
    width: int
    height: int
    has_jxl: bool  # True if DNG contains JPEG XL compression (iPhone ProRAW)

    @classmethod
    def from_exiftool(cls, path: Path, data: dict[str, str | int | None]) -> Self:
        """Factory method to parse exiftool JSON output."""
        # Parse dimensions
        width = data.get("ImageWidth", 0)
        height = data.get("ImageHeight", 0)
        if isinstance(width, str):
            width = int(width) if width.isdigit() else 0
        if isinstance(height, str):
            height = int(height) if height.isdigit() else 0

        # Check for JXL compression
        compression = data.get("Compression", "")
        has_jxl = isinstance(compression, str) and "jpeg xl" in compression.lower()

        return cls(
            path=path,
            width=int(width) if width else 0,
            height=int(height) if height else 0,
            has_jxl=has_jxl,
        )


# ═══════════════════════════════════════════════════════════════════
#                        RUNTIME HELPERS
# ═══════════════════════════════════════════════════════════════════


def _detect_homebrew_prefix() -> Path | None:
    """Detect Homebrew prefix if available."""
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
    """Build LD_LIBRARY_PATH for local binaries.

    Libraries:
    - ./build/lib: libraw
    - ./build/deps/lib or lib64: libjxl, libbrotli, etc.
    - ./avif_build/lib: libavif (if present)
    - Homebrew LLVM: libc++, libomp (clang runtime)
    """
    paths: list[str] = []

    # LibRaw libraries
    build_lib = SCRIPT_DIR / "build" / "lib"
    if build_lib.exists():
        paths.append(str(build_lib))

    # libjxl and dependencies (try both lib and lib64)
    deps_lib = SCRIPT_DIR / "build" / "deps" / "lib"
    if deps_lib.exists():
        paths.append(str(deps_lib))
    deps_lib64 = SCRIPT_DIR / "build" / "deps" / "lib64"
    if deps_lib64.exists():
        paths.append(str(deps_lib64))

    # libavif (if built with local libs)
    avif_lib = SCRIPT_DIR / "avif_build" / "lib"
    if avif_lib.exists():
        paths.append(str(avif_lib))

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


# ═══════════════════════════════════════════════════════════════════
#                        DNG METADATA EXTRACTION
# ═══════════════════════════════════════════════════════════════════


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
        return 0.0


def _extract_profile_tone_curve(
    dng_path: Path,
) -> tuple[NDArray[np.float32], NDArray[np.float32]] | None:
    """Extract ProfileToneCurve from DNG for global tone mapping.

    ProfileToneCurve is a 1D lookup table (257 input/output pairs) that applies
    a contrast curve to the image. iPhone ProRAW typically includes an S-curve
    that darkens shadows and adds contrast.

    Returns:
        Tuple of (input_values, output_values) as numpy arrays, or None if not found.
    """
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

        def search_ifd(
            offset: int,
        ) -> tuple[NDArray[np.float32], NDArray[np.float32]] | None:
            if offset < 2 or offset >= len(data) - 2:
                return None

            num_entries = struct.unpack(f"{fmt}H", data[offset : offset + 2])[0]
            if num_entries > 1000:
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


def _extract_profile_gain_table_map(dng_path: Path) -> PGTMData | None:
    """Extract ProfileGainTableMap (PGTM) from DNG for local tone mapping.

    DNG 1.6 introduces ProfileGainTableMap - a 3D lookup table for spatially-varying
    local tone mapping. iPhone ProRAW uses ~197KB PGTM data per image.

    Returns:
        PGTMData TypedDict with PGTM parameters and table, or None if not present.
    """
    try:
        result = subprocess.run(
            ["exiftool", "-b", "-ProfileGainTableMap", str(dng_path)],
            capture_output=True,
            check=True,
        )
        data = result.stdout
        if len(data) < 64:
            return None

        # Parse header (BIG-ENDIAN format) per DNG SDK
        points_v = struct.unpack_from(">I", data, 0)[0]
        points_h = struct.unpack_from(">I", data, 4)[0]
        spacing_v = struct.unpack_from(">d", data, 8)[0]
        spacing_h = struct.unpack_from(">d", data, 16)[0]
        origin_v = struct.unpack_from(">d", data, 24)[0]
        origin_h = struct.unpack_from(">d", data, 32)[0]
        num_table_points = struct.unpack_from(">I", data, 40)[0]
        input_weights = struct.unpack_from(">5f", data, 44)

        # Table starts at offset 64
        table_size = points_v * points_h * num_table_points
        expected_data_len = 64 + table_size * 4
        if len(data) < expected_data_len:
            return None

        table = np.frombuffer(data, dtype=">f4", count=table_size, offset=64)
        table = table.reshape(points_v, points_h, num_table_points).astype(np.float32)

        return PGTMData(
            points_v=points_v,
            points_h=points_h,
            spacing_v=spacing_v,
            spacing_h=spacing_h,
            origin_v=origin_v,
            origin_h=origin_h,
            num_table_points=num_table_points,
            input_weights=(
                input_weights[0],
                input_weights[1],
                input_weights[2],
                input_weights[3],
                input_weights[4],
            ),
            table=table,
        )
    except (subprocess.CalledProcessError, ValueError):
        return None


# ═══════════════════════════════════════════════════════════════════
#                        DNG SPLINE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════


def _solve_dng_spline(
    x_points: NDArray[np.float64],
    y_points: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute C2-continuous spline slopes using DNG SDK algorithm.

    Implements dng_spline_solver::Solve() from dng_spline.cpp:57-145.
    """
    count = len(x_points)
    S = np.zeros(count, dtype=np.float64)

    A = x_points[1] - x_points[0]
    B = (y_points[1] - y_points[0]) / A
    S[0] = B

    for j in range(2, count):
        C = x_points[j] - x_points[j - 1]
        D = (y_points[j] - y_points[j - 1]) / C
        S[j - 1] = (B * C + D * A) / (A + C)
        A, B = C, D

    S[count - 1] = 2.0 * B - S[count - 2]
    S[0] = 2.0 * S[0] - S[1]

    if count > 2:
        E = np.zeros(count, dtype=np.float64)
        F = np.zeros(count, dtype=np.float64)
        G = np.zeros(count, dtype=np.float64)

        F[0] = 0.5
        E[count - 1] = 0.5
        G[0] = 0.75 * (S[0] + S[1])
        G[count - 1] = 0.75 * (S[count - 2] + S[count - 1])

        for j in range(1, count - 1):
            A = (x_points[j + 1] - x_points[j - 1]) * 2.0
            E[j] = (x_points[j + 1] - x_points[j]) / A
            F[j] = (x_points[j] - x_points[j - 1]) / A
            G[j] = 1.5 * S[j]

        for j in range(1, count):
            A = 1.0 - F[j - 1] * E[j]
            if j != count - 1:
                F[j] /= A
            G[j] = (G[j] - G[j - 1] * E[j]) / A

        for j in range(count - 2, -1, -1):
            G[j] = G[j] - F[j] * G[j + 1]

        S = G

    return S


def _evaluate_spline_segment(
    x: NDArray[np.float64],
    x0: float,
    y0: float,
    s0: float,
    x1: float,
    y1: float,
    s1: float,
) -> NDArray[np.float64]:
    """Evaluate cubic Hermite spline segment."""
    A = x1 - x0
    B = (x - x0) / A
    C = (x1 - x) / A

    D = ((y0 * (2.0 - C + B) + (s0 * A * B)) * (C * C)) + (
        (y1 * (2.0 - B + C) - (s1 * A * C)) * (B * B)
    )

    return D


# ═══════════════════════════════════════════════════════════════════
#                        OVERRANGE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════


def _encode_overrange(x: NDArray[np.float32]) -> NDArray[np.float32]:
    """Compress HDR values for table lookup. Maps [0, inf) -> [0, 1)."""
    x = np.maximum(x, 0.0)
    return (x * (256.0 + x) / (256.0 * (1.0 + x))).astype(np.float32)


def _decode_overrange(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Expand encoded values back to HDR."""
    x = np.maximum(x, 0.0)
    return 16.0 * ((8.0 * x) - 8.0 + np.sqrt(64.0 * x * x - 127.0 * x + 64.0))


# ═══════════════════════════════════════════════════════════════════
#                        TONE CURVE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════


def _apply_profile_tone_curve(
    img: NDArray[np.float32],
    curve_inputs: NDArray[np.float32],
    curve_outputs: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Apply ProfileToneCurve using DNG SDK's RGBTone algorithm (HDR path).

    Uses EncodeOverrange/DecodeOverrange to preserve HDR values > 1.0.
    """
    slopes = _solve_dng_spline(
        curve_inputs.astype(np.float64), curve_outputs.astype(np.float64)
    )

    def apply_curve(values: NDArray[np.float32]) -> NDArray[np.float64]:
        result = np.zeros_like(values, dtype=np.float64)
        encoded = _encode_overrange(values)

        indices = np.searchsorted(curve_inputs, encoded, side="right")
        indices = np.clip(indices, 1, len(curve_inputs) - 1)

        for j in range(1, len(curve_inputs)):
            mask = indices == j
            if not np.any(mask):
                continue

            x_vals = encoded[mask]
            x0, y0, s0 = curve_inputs[j - 1], curve_outputs[j - 1], slopes[j - 1]
            x1, y1, s1 = curve_inputs[j], curve_outputs[j], slopes[j]

            result[mask] = _evaluate_spline_segment(x_vals, x0, y0, s0, x1, y1, s1)

        below_mask = encoded <= curve_inputs[0]
        result[below_mask] = curve_outputs[0]

        result = _decode_overrange(result)
        return result

    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    rgb_stack = np.stack([r, g, b], axis=-1)
    max_val = np.max(rgb_stack, axis=-1)
    min_val = np.min(rgb_stack, axis=-1)

    max_curved = apply_curve(max_val)
    min_curved = apply_curve(min_val)

    result = np.zeros_like(img)
    gray_mask = (max_val - min_val) < 1e-10

    for i, channel in enumerate([r, g, b]):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(
                gray_mask, 0.5, (channel - min_val) / (max_val - min_val + 1e-10)
            )
            t = np.clip(t, 0.0, 1.0)
        result[:, :, i] = min_curved + t * (max_curved - min_curved)

    return result.astype(np.float32)


def _apply_profile_tone_curve_sdr(
    img: NDArray[np.float32],
    curve_inputs: NDArray[np.float32],
    curve_outputs: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Apply ProfileToneCurve for SDR (no overrange encode/decode).

    Identical to _apply_profile_tone_curve() except:
    - Input is clipped to [0, 1] instead of encoded with _encode_overrange()
    - Output is returned directly instead of decoded with _decode_overrange()

    This matches DNG SDK behavior when SupportOverrange=false.
    """
    slopes = _solve_dng_spline(
        curve_inputs.astype(np.float64), curve_outputs.astype(np.float64)
    )

    def apply_curve(values: NDArray[np.float32]) -> NDArray[np.float64]:
        result = np.zeros_like(values, dtype=np.float64)
        clipped = np.clip(values, 0.0, 1.0)

        indices = np.searchsorted(curve_inputs, clipped, side="right")
        indices = np.clip(indices, 1, len(curve_inputs) - 1)

        for j in range(1, len(curve_inputs)):
            mask = indices == j
            if not np.any(mask):
                continue
            x_vals = clipped[mask]
            x0, y0, s0 = curve_inputs[j - 1], curve_outputs[j - 1], slopes[j - 1]
            x1, y1, s1 = curve_inputs[j], curve_outputs[j], slopes[j]
            result[mask] = _evaluate_spline_segment(x_vals, x0, y0, s0, x1, y1, s1)

        below_mask = clipped <= curve_inputs[0]
        result[below_mask] = curve_outputs[0]

        return result

    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    rgb_stack = np.stack([r, g, b], axis=-1)
    max_val = np.max(rgb_stack, axis=-1)
    min_val = np.min(rgb_stack, axis=-1)

    max_curved = apply_curve(max_val)
    min_curved = apply_curve(min_val)

    result = np.zeros_like(img)
    gray_mask = (max_val - min_val) < 1e-10

    for i, channel in enumerate([r, g, b]):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(
                gray_mask, 0.5, (channel - min_val) / (max_val - min_val + 1e-10)
            )
            t = np.clip(t, 0.0, 1.0)
        result[:, :, i] = min_curved + t * (max_curved - min_curved)

    return result.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
#                        PGTM PROCESSING
# ═══════════════════════════════════════════════════════════════════


def _compute_pgtm_gains(
    img: NDArray[np.float32],
    pgtm: PGTMData,
    baseline_exposure_ev: float = 0.0,
) -> NDArray[np.float32]:
    """Compute PGTM gains for a numpy array.

    Per DNG SDK, the weight is computed from RAW pixel values and then
    scaled by 2^BaselineExposure before table lookup.
    """
    height, width = img.shape[:2]

    points_v = pgtm["points_v"]
    points_h = pgtm["points_h"]
    num_table_points = pgtm["num_table_points"]
    spacing_v = pgtm["spacing_v"]
    spacing_h = pgtm["spacing_h"]
    origin_v = pgtm["origin_v"]
    origin_h = pgtm["origin_h"]
    input_weights = np.array(pgtm["input_weights"])
    table = pgtm["table"]

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

    if baseline_exposure_ev != 0.0:
        exposure_weight_gain = 2.0**baseline_exposure_ev
        weight = weight * exposure_weight_gain

    weight = np.clip(weight, 0.0, 1.0)
    weight_idx = weight * num_table_points
    weight_idx = np.clip(weight_idx, 0, num_table_points - 1)

    rows = np.arange(height, dtype=np.float32) + 0.5
    cols = np.arange(width, dtype=np.float32) + 0.5
    v_image = rows / height
    u_image = cols / width
    row_coords = np.clip((v_image - origin_v) / spacing_v, 0, points_v - 1)
    col_coords = np.clip((u_image - origin_h) / spacing_h, 0, points_h - 1)
    row_grid, col_grid = np.meshgrid(row_coords, col_coords, indexing="ij")

    # Trilinear interpolation
    v_floor = np.floor(row_grid).astype(np.int32)
    v_ceil = np.minimum(v_floor + 1, points_v - 1)
    v_frac = row_grid - v_floor
    v_floor = np.clip(v_floor, 0, points_v - 1)

    h_floor = np.floor(col_grid).astype(np.int32)
    h_ceil = np.minimum(h_floor + 1, points_h - 1)
    h_frac = col_grid - h_floor
    h_floor = np.clip(h_floor, 0, points_h - 1)

    n_floor = np.floor(weight_idx).astype(np.int32)
    n_ceil = np.minimum(n_floor + 1, num_table_points - 1)
    n_frac = weight_idx - n_floor
    n_floor = np.clip(n_floor, 0, num_table_points - 1)

    c000 = table[v_floor, h_floor, n_floor]
    c001 = table[v_floor, h_floor, n_ceil]
    c010 = table[v_floor, h_ceil, n_floor]
    c011 = table[v_floor, h_ceil, n_ceil]
    c100 = table[v_ceil, h_floor, n_floor]
    c101 = table[v_ceil, h_floor, n_ceil]
    c110 = table[v_ceil, h_ceil, n_floor]
    c111 = table[v_ceil, h_ceil, n_ceil]

    c00 = c000 * (1 - n_frac) + c001 * n_frac
    c01 = c010 * (1 - n_frac) + c011 * n_frac
    c10 = c100 * (1 - n_frac) + c101 * n_frac
    c11 = c110 * (1 - n_frac) + c111 * n_frac

    c0 = c00 * (1 - h_frac) + c01 * h_frac
    c1 = c10 * (1 - h_frac) + c11 * h_frac

    gains = c0 * (1 - v_frac) + c1 * v_frac

    return gains.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
#                        EXPOSURE RAMP
# ═══════════════════════════════════════════════════════════════════


def _exposure_ramp(
    x: NDArray[np.float32],
    white: float,
    black: float = 0.0,
    support_overrange: bool = True,
) -> NDArray[np.float32]:
    """Apply DNG SDK exposure ramp with quadratic shadow transition.

    Implements dng_function_exposure_ramp from dng_render.cpp:50-103.
    """
    slope = 1.0 / (white - black) if white > black else 1.0

    kMaxCurveX = 0.5
    kMaxCurveY = 1.0 / 16.0

    radius = min(kMaxCurveX * black, kMaxCurveY / slope) if black > 0 else 0.0
    qscale = slope / (4.0 * radius) if radius > 0 else 0.0

    result = np.zeros_like(x, dtype=np.float32)

    mask1 = x <= (black - radius)
    result[mask1] = 0.0

    mask2 = x >= (black + radius)
    y = (x[mask2] - black) * slope
    if not support_overrange:
        y = np.minimum(y, 1.0)
    result[mask2] = y

    mask3 = ~mask1 & ~mask2
    if np.any(mask3):
        y = x[mask3] - (black - radius)
        result[mask3] = qscale * y * y

    return result


# ═══════════════════════════════════════════════════════════════════
#                        COLOR CONVERSION
# ═══════════════════════════════════════════════════════════════════


def _convert_prophoto_to_rec2020(img: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert linear ProPhoto RGB D65 to linear Rec.2020 with gamut mapping."""
    matrix = np.array(
        [
            [0.8198343, 0.0263074, 0.1538522],
            [0.0453679, 0.9474589, 0.0071759],
            [-0.0006235, 0.0377712, 0.9628510],
        ],
        dtype=np.float32,
    )

    lum_coeffs = np.array([0.2627, 0.6780, 0.0593], dtype=np.float32)

    h, w, c = img.shape
    flat = img.reshape(-1, 3)
    converted = flat @ matrix.T

    min_vals = np.min(converted, axis=1)
    out_of_gamut = min_vals < 0

    if np.any(out_of_gamut):
        lum = (converted[out_of_gamut] @ lum_coeffs)[:, np.newaxis]
        rgb_oog = converted[out_of_gamut]

        with np.errstate(divide="ignore", invalid="ignore"):
            diff = lum - rgb_oog
            t = np.where(diff > 0, lum / diff, 1.0)
            t_max = np.min(np.where(rgb_oog < 0, t, 1.0), axis=1, keepdims=True)

        converted[out_of_gamut] = lum + t_max * (rgb_oog - lum)
        converted[out_of_gamut] = np.maximum(converted[out_of_gamut], 0.0)

    return converted.reshape(h, w, 3).astype(np.float32)


def _convert_prophoto_to_display_p3(img: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert linear ProPhoto RGB D65 to linear Display P3 with gamut mapping.

    Matrix computed from CIE 1931 chromaticity coordinates:
    - ProPhoto RGB primaries (D65 adapted): R(0.7347,0.2653) G(0.1596,0.8404) B(0.0366,0.0001)
    - Display P3 primaries (D65): R(0.680,0.320) G(0.265,0.690) B(0.150,0.060)
    - White point: D65 (0.3127, 0.3290)

    Validated: white (1,1,1) -> (1,1,1), row sums = 1.0
    """
    matrix = np.array(
        [
            [1.6656413, -0.3301368, -0.3355045],
            [-0.1490164, 1.1574112, -0.0083947],
            [0.0064396, -0.0500168, 1.0435771],
        ],
        dtype=np.float32,
    )

    # Display P3 luminance coefficients (Y row of RGB->XYZ matrix)
    lum_coeffs = np.array([0.228975, 0.691739, 0.079287], dtype=np.float32)

    h, w, c = img.shape
    flat = img.reshape(-1, 3)
    converted = flat @ matrix.T

    min_vals = np.min(converted, axis=1)
    out_of_gamut = min_vals < 0

    if np.any(out_of_gamut):
        lum = (converted[out_of_gamut] @ lum_coeffs)[:, np.newaxis]
        rgb_oog = converted[out_of_gamut]

        with np.errstate(divide="ignore", invalid="ignore"):
            diff = lum - rgb_oog
            t = np.where(diff > 0, lum / diff, 1.0)
            t_max = np.min(np.where(rgb_oog < 0, t, 1.0), axis=1, keepdims=True)

        converted[out_of_gamut] = lum + t_max * (rgb_oog - lum)
        converted[out_of_gamut] = np.maximum(converted[out_of_gamut], 0.0)

    return converted.reshape(h, w, 3).astype(np.float32)


def _convert_rec2020_to_display_p3(img: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert linear Rec.2020 to linear Display P3 with gamut mapping.

    Matrix computed from:
    - Rec.2020 primaries: R(0.708,0.292) G(0.170,0.797) B(0.131,0.046)
    - Display P3 primaries: R(0.680,0.320) G(0.265,0.690) B(0.150,0.060)
    - White point: D65 (0.3127, 0.3290)

    This is the inverse of P3→Rec.2020 matrix. Row sums = 1.0 for white preservation.
    """
    matrix = np.array(
        [
            [1.22494018, -0.22494018, 0.0],
            [-0.04205695, 1.04205695, 0.0],
            [-0.01963755, -0.07863605, 1.0982736],
        ],
        dtype=np.float32,
    )

    # Display P3 luminance coefficients (Y row of RGB->XYZ matrix)
    lum_coeffs = np.array([0.228975, 0.691739, 0.079287], dtype=np.float32)

    h, w, c = img.shape
    flat = img.reshape(-1, 3)
    converted = flat @ matrix.T

    min_vals = np.min(converted, axis=1)
    out_of_gamut = min_vals < 0

    if np.any(out_of_gamut):
        lum = (converted[out_of_gamut] @ lum_coeffs)[:, np.newaxis]
        rgb_oog = converted[out_of_gamut]

        with np.errstate(divide="ignore", invalid="ignore"):
            diff = lum - rgb_oog
            t = np.where(diff > 0, lum / diff, 1.0)
            t_max = np.min(np.where(rgb_oog < 0, t, 1.0), axis=1, keepdims=True)

        converted[out_of_gamut] = lum + t_max * (rgb_oog - lum)
        converted[out_of_gamut] = np.maximum(converted[out_of_gamut], 0.0)

    return converted.reshape(h, w, 3).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
#                        TRANSFER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════


def _apply_srgb_gamma(linear: NDArray[np.float32]) -> NDArray[np.float32]:
    """Apply sRGB transfer function (IEC 61966-2-1).

    Piecewise:
        x <= 0.0031308: 12.92 * x
        x > 0.0031308:  1.055 * x^(1/2.4) - 0.055
    """
    linear = np.clip(linear, 0.0, 1.0)
    threshold = 0.0031308
    return np.where(
        linear <= threshold,
        12.92 * linear,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    ).astype(np.float32)


def _apply_pq_oetf(linear_nits: NDArray[np.float32]) -> NDArray[np.float32]:
    """Apply PQ (SMPTE ST 2084) OETF."""
    m1 = 0.1593017578125
    m2 = 78.84375
    c1 = 0.8359375
    c2 = 18.8515625
    c3 = 18.6875

    L = np.clip(linear_nits / 10000.0, 0.0, 1.0)
    L_m1 = np.power(L, m1)
    numerator = c1 + c2 * L_m1
    denominator = 1.0 + c3 * L_m1
    pq = np.power(numerator / denominator, m2)

    return pq.astype(np.float32)


def _scale_to_nits(
    linear_normalized: NDArray[np.float32], sdr_white_nits: float = 203.0
) -> NDArray[np.float32]:
    """Scale normalized linear values to absolute nits for PQ encoding."""
    return (linear_normalized * sdr_white_nits).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
#                        PPM I/O
# ═══════════════════════════════════════════════════════════════════


def _read_ppm_as_float(ppm_path: Path) -> tuple[NDArray[np.float32], int, int]:
    """Read 16-bit PPM as normalized float32 array."""
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


def _write_png_from_float(
    png_path: Path, img: NDArray[np.float32]
) -> None:
    """Write normalized float32 array as 16-bit PNG using pypng.

    Lossless conversion - PNG compression preserves all data.
    16-bit depth ensures full precision for avifenc to quantize.
    """
    # Convert to 16-bit unsigned integer
    img_16bit = np.clip(img * 65535.0, 0, 65535).astype(np.uint16)
    height, width = img_16bit.shape[:2]
    # pypng Writer for 16-bit RGB
    writer = png.Writer(width=width, height=height, bitdepth=16, greyscale=False)
    # pypng expects rows as (H, W*3) array - flatten RGB channels per row
    rows = img_16bit.reshape(height, width * 3)
    with open(png_path, "wb") as f:
        writer.write(f, rows)


# ═══════════════════════════════════════════════════════════════════
#                        VALIDATION
# ═══════════════════════════════════════════════════════════════════


def validate_environment(source_dir: Path, dest_dir: Path) -> None:
    """Validate required tools and directories."""
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = _get_library_path()

    # 1. Check exiftool (metadata transfer)
    if not shutil.which("exiftool"):
        raise ValidationError("exiftool not found in PATH")

    # 2. Check dcraw_emu_dng (DNG processing)
    if not DCRAW_EMU.exists():
        raise ValidationError(
            f"dcraw_emu_dng not found at {DCRAW_EMU}. Run libraw_dng.py first."
        )
    try:
        result = subprocess.run(
            [str(DCRAW_EMU)], capture_output=True, text=True, env=env, timeout=5
        )
        if "cannot open shared object file" in result.stderr:
            raise ValidationError(f"dcraw_emu_dng missing libraries: {result.stderr}")
    except subprocess.TimeoutExpired:
        raise ValidationError("dcraw_emu_dng timed out during validation")

    # 3. Check avifenc (AVIF encoding)
    if not AVIFENC.exists():
        raise ValidationError(
            f"avifenc not found at {AVIFENC}. Run avif_tools.py first."
        )
    try:
        result = subprocess.run(
            [str(AVIFENC), "--version"],
            capture_output=True,
            text=True,
            env=env,
            timeout=5,
        )
        if result.returncode != 0:
            raise ValidationError(f"avifenc failed: {result.stderr}")
        if "cannot open shared object file" in result.stderr:
            raise ValidationError(f"avifenc missing libraries: {result.stderr}")
    except subprocess.TimeoutExpired:
        raise ValidationError("avifenc timed out during validation")

    # 4. Check avifgainmaputil (gain map creation)
    if not AVIFGAINMAPUTIL.exists():
        raise ValidationError(
            f"avifgainmaputil not found at {AVIFGAINMAPUTIL}. Run avif_tools.py first."
        )
    try:
        result = subprocess.run(
            [str(AVIFGAINMAPUTIL), "help"],
            capture_output=True,
            text=True,
            env=env,
            timeout=5,
        )
        if "cannot open shared object file" in result.stderr:
            raise ValidationError(
                f"avifgainmaputil missing libraries: {result.stderr}"
            )
        if result.returncode != 0:
            raise ValidationError(f"avifgainmaputil failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        raise ValidationError("avifgainmaputil timed out during validation")

    # 5. Check directories
    if not source_dir.is_dir():
        raise ValidationError(f"Source not a directory: {source_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
#                        IMAGE SCANNING
# ═══════════════════════════════════════════════════════════════════


def scan_images(source_dir: Path) -> tuple[Path, ...]:
    """Scan directory for DNG files."""
    images = [
        f
        for f in source_dir.iterdir()
        if f.is_file() and f.suffix.lower() == ".dng"
    ]
    return tuple(sorted(images, key=lambda p: p.name.lower()))


def probe_image(path: Path) -> ImageInfo:
    """Extract DNG metadata using exiftool."""
    try:
        result = subprocess.run(
            [
                "exiftool",
                "-j",
                "-Compression",
                "-ImageWidth",
                "-ImageHeight",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(result.stdout)
        if not data:
            raise DngProcessingError(f"No metadata found for {path}")
        return ImageInfo.from_exiftool(path, data[0])
    except subprocess.CalledProcessError as e:
        raise DngProcessingError(f"exiftool failed for {path}: {e.stderr}")
    except json.JSONDecodeError as e:
        raise DngProcessingError(f"Failed to parse exiftool output for {path}: {e}")


def should_process_image(info: ImageInfo, dest_dir: Path, force: bool = False) -> bool:
    """Check if image should be processed."""
    if not info.has_jxl:
        return False  # Skip non-iPhone ProRAW DNGs
    output_path = dest_dir / f"{info.path.stem}.avif"
    if output_path.exists() and not force:
        return False
    return True


# ═══════════════════════════════════════════════════════════════════
#                        METADATA TRANSFER
# ═══════════════════════════════════════════════════════════════════


def _transfer_metadata(dng_path: Path, avif_path: Path) -> None:
    """Copy metadata from DNG, excluding color tags."""
    try:
        subprocess.run(
            [
                "exiftool",
                "-tagsFromFile",
                str(dng_path),
                "-all:all",
                "-overwrite_original",
                # Exclude CICP/color tags set by avifgainmaputil
                "-ColorPrimaries=",
                "-TransferCharacteristics=",
                "-MatrixCoefficients=",
                "-ColorSpace=",
                str(avif_path),
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        raise MetadataError(f"Failed to transfer metadata: {e.stderr}")


# ═══════════════════════════════════════════════════════════════════
#                        MAIN PROCESSING
# ═══════════════════════════════════════════════════════════════════


def process_dng_to_avif(
    info: ImageInfo,
    dest_dir: Path,
    quality: int = 100,
    qgain_map: int = 90,
    speed: int = 3,
    depth: int = 10,
) -> Path:
    """Process iPhone ProRAW DNG to gain map AVIF.

    Creates SDR base layer + HDR alternate layer with ISO 21496-1 gain map.
    """
    output_path = dest_dir / f"{info.path.stem}.avif"

    # Temp files (hidden, cleaned up in finally)
    # Dual decode: SDR with clipped highlights, HDR with preserved highlights
    temp_linear_ppm_sdr = dest_dir / f".{info.path.stem}_linear_sdr.ppm"
    temp_linear_ppm_hdr = dest_dir / f".{info.path.stem}_linear_hdr.ppm"
    temp_sdr_png = dest_dir / f".{info.path.stem}_sdr.png"  # For avifenc (PNG)
    temp_hdr_png = dest_dir / f".{info.path.stem}_hdr.png"  # For avifenc (PNG)
    temp_sdr_avif = dest_dir / f".{info.path.stem}_sdr.avif"
    temp_hdr_avif = dest_dir / f".{info.path.stem}_hdr.avif"

    temp_files = [
        temp_linear_ppm_sdr,
        temp_linear_ppm_hdr,
        temp_sdr_png,
        temp_hdr_png,
        temp_sdr_avif,
        temp_hdr_avif,
    ]

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = _get_library_path()

    try:
        # ═══════════════════════════════════════════════════════════
        # Phase 1: Extract DNG metadata
        # ═══════════════════════════════════════════════════════════

        baseline_exposure_ev = _get_baseline_exposure_ev(info.path)
        pgtm = _extract_profile_gain_table_map(info.path)
        tone_curve = _extract_profile_tone_curve(info.path)

        # Compute exposure parameters
        exposure_white = 1.0 / (2.0 ** max(0.0, baseline_exposure_ev))

        # ═══════════════════════════════════════════════════════════
        # Phase 2: Dual Decode - SDR and HDR with different highlight handling
        # ═══════════════════════════════════════════════════════════

        # SDR decode: -H 0 clips highlights (standard SDR rendering)
        # -M disables embedded color matrix (ProRAW is already Rec.2020)
        # -o 8 outputs Rec.2020 colorspace directly
        dcraw_cmd_sdr = [
            str(DCRAW_EMU),
            "-W",  # Disable auto-brightness
            "-g", "1", "1",  # Linear output
            "-dngsdk",  # Use DNG SDK for JXL decoding
            "-6",  # 16-bit output
            "-q", "3",  # AHD interpolation
            "-H", "0",  # Clip highlights for SDR
            "-M",  # Disable embedded color matrix (ProRAW is already Rec.2020)
            "-o", "8",  # Output Rec.2020 (not ProPhoto)
            "-Z", str(temp_linear_ppm_sdr),
            str(info.path),
        ]

        # HDR decode: -H 1 preserves highlights for HDR headroom
        dcraw_cmd_hdr = [
            str(DCRAW_EMU),
            "-W",  # Disable auto-brightness
            "-g", "1", "1",  # Linear output
            "-dngsdk",  # Use DNG SDK for JXL decoding
            "-6",  # 16-bit output
            "-q", "3",  # AHD interpolation
            "-H", "1",  # Unclip highlights for HDR
            "-M",  # Disable embedded color matrix (ProRAW is already Rec.2020)
            "-o", "8",  # Output Rec.2020 (not ProPhoto)
            "-Z", str(temp_linear_ppm_hdr),
            str(info.path),
        ]

        # Run both decodes
        result_sdr = subprocess.run(dcraw_cmd_sdr, capture_output=True, text=True, env=env)
        if result_sdr.returncode != 0:
            raise DngProcessingError(
                f"dcraw_emu_dng (SDR) failed: {result_sdr.stderr or result_sdr.stdout}"
            )

        result_hdr = subprocess.run(dcraw_cmd_hdr, capture_output=True, text=True, env=env)
        if result_hdr.returncode != 0:
            raise DngProcessingError(
                f"dcraw_emu_dng (HDR) failed: {result_hdr.stderr or result_hdr.stdout}"
            )

        if not temp_linear_ppm_sdr.exists():
            raise DngProcessingError(f"dcraw_emu_dng produced no SDR output for {info.path.name}")
        if not temp_linear_ppm_hdr.exists():
            raise DngProcessingError(f"dcraw_emu_dng produced no HDR output for {info.path.name}")

        # ═══════════════════════════════════════════════════════════
        # Phase 3: SDR Branch (full tone mapping pipeline)
        # Data is Rec.2020 linear (ProRAW native colorspace)
        # ═══════════════════════════════════════════════════════════

        # Read SDR linear data (Rec.2020, highlights clipped via -H 0)
        dcraw_linear_sdr, width, height = _read_ppm_as_float(temp_linear_ppm_sdr)

        # Apply PGTM for SDR (local tone mapping for SDR contrast)
        if pgtm:
            pgtm_gains = _compute_pgtm_gains(dcraw_linear_sdr, pgtm, baseline_exposure_ev)
            sdr_pgtm_output = dcraw_linear_sdr * pgtm_gains[:, :, np.newaxis]
        else:
            sdr_pgtm_output = dcraw_linear_sdr.copy()

        # Exposure ramp with clipping (SDR path)
        sdr_linear = _exposure_ramp(
            sdr_pgtm_output,
            white=exposure_white,
            black=0.0,
            support_overrange=False,
        )

        # Apply tone curve (SDR version - no overrange)
        if tone_curve:
            curve_inputs, curve_outputs = tone_curve
            sdr_linear = _apply_profile_tone_curve_sdr(
                sdr_linear, curve_inputs, curve_outputs
            )

        # Convert Rec.2020 → Display P3 (data is already Rec.2020 from dcraw)
        sdr_p3 = _convert_rec2020_to_display_p3(sdr_linear)

        # Apply sRGB gamma
        sdr_encoded = _apply_srgb_gamma(sdr_p3)

        # Write SDR PNG (16-bit lossless)
        _write_png_from_float(temp_sdr_png, sdr_encoded)

        # ═══════════════════════════════════════════════════════════
        # Phase 4: HDR Branch (simplified - preserve full dynamic range)
        # Data is Rec.2020 linear (ProRAW native colorspace) - no conversion needed
        # ═══════════════════════════════════════════════════════════

        # Read HDR linear data (Rec.2020, highlights preserved via -H 1)
        dcraw_linear_hdr, _, _ = _read_ppm_as_float(temp_linear_ppm_hdr)

        # Skip PGTM for HDR - it's designed for SDR local contrast
        # Apply exposure compensation directly to preserve HDR headroom
        exposure_gain = 2.0 ** baseline_exposure_ev
        hdr_linear = dcraw_linear_hdr * exposure_gain

        # Skip tone curve for HDR - preserve linear response in highlights
        # The S-curve would compress our HDR headroom

        # Data is already Rec.2020 linear from dcraw (no conversion needed)
        hdr_rec2020 = hdr_linear

        # Scale to nits and apply PQ
        # Values > 1.0 will exceed 203 nits, giving proper HDR headroom
        hdr_nits = _scale_to_nits(hdr_rec2020, sdr_white_nits=203.0)
        hdr_pq = _apply_pq_oetf(hdr_nits)

        # Write HDR PNG (16-bit lossless)
        _write_png_from_float(temp_hdr_png, hdr_pq)

        # ═══════════════════════════════════════════════════════════
        # Phase 5: Encode AVIFs
        # ═══════════════════════════════════════════════════════════

        # Encode SDR AVIF (8-bit, CICP 12/13/0 = Display P3 + sRGB)
        sdr_cmd = [
            str(AVIFENC),
            "--depth",
            "8",
            "--yuv",
            "444",
            "--cicp",
            "12/13/0",
            "-q",
            str(quality),
            "-s",
            str(speed),
            "--range",
            "full",
            "--ignore-exif",
            "--ignore-xmp",
            "--ignore-icc",
            str(temp_sdr_png),
            str(temp_sdr_avif),
        ]

        result = subprocess.run(sdr_cmd, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            raise AvifEncodingError(f"SDR avifenc failed: {result.stderr or result.stdout}")

        # Encode HDR AVIF (10-bit, CICP 9/16/9 = BT.2020 + PQ + BT.2020 matrix)
        hdr_cmd = [
            str(AVIFENC),
            "--depth",
            str(depth),
            "--yuv",
            "444",
            "--cicp",
            "9/16/9",
            "-q",
            str(quality),
            "-s",
            str(speed),
            "--range",
            "full",
            "--ignore-exif",
            "--ignore-xmp",
            "--ignore-icc",
            str(temp_hdr_png),
            str(temp_hdr_avif),
        ]

        result = subprocess.run(hdr_cmd, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            raise AvifEncodingError(f"HDR avifenc failed: {result.stderr or result.stdout}")

        # ═══════════════════════════════════════════════════════════
        # Phase 6: Combine with gain map
        # ═══════════════════════════════════════════════════════════

        combine_cmd = [
            str(AVIFGAINMAPUTIL),
            "combine",
            str(temp_sdr_avif),
            str(temp_hdr_avif),
            str(output_path),
            "-q",
            str(quality),
            "--qgain-map",
            str(qgain_map),
            "-s",
            str(speed),
            "-d",
            str(depth),
            "-y",
            "444",
            "--depth-gain-map",
            "10",
            "--yuv-gain-map",
            "444",
            "--ignore-profile",
            "--cicp-base",
            "12/13/0",
            "--cicp-alternate",
            "9/16/9",
        ]

        result = subprocess.run(combine_cmd, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            raise AvifEncodingError(f"avifgainmaputil failed: {result.stderr or result.stdout}")

        # ═══════════════════════════════════════════════════════════
        # Phase 7: Transfer metadata
        # ═══════════════════════════════════════════════════════════

        _transfer_metadata(info.path, output_path)

        # Copy filesystem timestamps
        shutil.copystat(info.path, output_path)

        return output_path

    finally:
        # Clean up temp files
        for temp_file in temp_files:
            if temp_file.exists():
                temp_file.unlink()


# ═══════════════════════════════════════════════════════════════════
#                        CLI
# ═══════════════════════════════════════════════════════════════════


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert iPhone ProRAW DNG files to gain map AVIF (ISO 21496-1).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output format:
  SDR base: Rec.2020 + sRGB (CICP 9/13/0)
  HDR alternate: Rec.2020 + PQ (CICP 9/16/0)

Examples:
  %(prog)s                          # Use default directories
  %(prog)s --source ~/Photos/raw    # Custom source directory
  %(prog)s -q 90 --qgain-map 80     # Adjust quality
  %(prog)s --debug                  # Keep intermediate files
""",
    )
    parser.add_argument(
        "--source",
        "-s",
        type=Path,
        default=Path(SOURCE_DIR),
        help=f"Source directory containing DNG files (default: {SOURCE_DIR})",
    )
    parser.add_argument(
        "--dest",
        "-d",
        type=Path,
        default=Path(DESTINATION_DIR),
        help=f"Destination directory for AVIF output (default: {DESTINATION_DIR})",
    )
    parser.add_argument(
        "--quality",
        "-q",
        type=int,
        default=100,
        help="Base AVIF quality 0-100 (default: 100)",
    )
    parser.add_argument(
        "--qgain-map",
        type=int,
        default=90,
        help="Gain map quality 0-100 (default: 90)",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=3,
        help="Encoder speed 0-10, lower=slower+better (default: 3)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=10,
        choices=[8, 10, 12],
        help="HDR output bit depth (default: 10)",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Reprocess files that already have output",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Keep intermediate PPM files for debugging",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="Directory for debug output (default: <dest>/debug/)",
    )
    return parser.parse_args()


def main() -> None:
    """Script entry point."""
    global DEBUG_MODE, DEBUG_OUTPUT_DIR

    args = parse_arguments()

    DEBUG_MODE = args.debug
    if DEBUG_MODE:
        DEBUG_OUTPUT_DIR = args.debug_dir or (args.dest / "debug")
        DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        console.print(f"[yellow]Debug mode enabled. Output: {DEBUG_OUTPUT_DIR}[/]\n")

    console.print("\n[bold]AVIF Gain Map Exporter[/] (DNG to ISO 21496-1 AVIF)\n")

    source_dir = args.source
    dest_dir = args.dest

    try:
        # Validate environment
        validate_environment(source_dir, dest_dir)

        # Scan for DNG files
        dng_paths = scan_images(source_dir)

        if not dng_paths:
            console.print(f"[yellow]No DNG files found in {source_dir}[/]")
            return

        console.print(f"Found {len(dng_paths)} DNG file(s) to analyze...\n")

        # Probe and filter images
        images_to_process: list[ImageInfo] = []
        skipped: list[tuple[str, str]] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing DNG files...", total=len(dng_paths))

            for path in dng_paths:
                try:
                    info = probe_image(path)
                    if should_process_image(info, dest_dir, args.force):
                        images_to_process.append(info)
                    elif not info.has_jxl:
                        skipped.append((path.name, "Not iPhone ProRAW"))
                    else:
                        skipped.append((path.name, "Output exists"))
                except DngProcessingError as e:
                    console.print(f"[red]Error probing {path.name}:[/] {e}")
                    skipped.append((path.name, "Probe failed"))

                progress.advance(task)

        # Display summary table
        table = Table(title="DNG Analysis Results")
        table.add_column("File", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Dimensions")
        table.add_column("Action")

        for info in images_to_process:
            table.add_row(
                info.path.name,
                "[green]Ready[/]",
                f"{info.width}x{info.height}",
                "[blue]Process[/]",
            )

        for name, reason in skipped:
            table.add_row(name, f"[dim]{reason}[/]", "-", "-")

        console.print(table)

        if not images_to_process:
            console.print("\n[yellow]No images to process.[/]")
            return

        # Process images
        console.print(f"\n[bold blue]Processing {len(images_to_process)} image(s)[/]\n")

        for i, info in enumerate(images_to_process, 1):
            try:
                output_path = process_dng_to_avif(
                    info,
                    dest_dir,
                    quality=args.quality,
                    qgain_map=args.qgain_map,
                    speed=args.speed,
                    depth=args.depth,
                )
                console.print(
                    f"[green]\u2713[/] [{i}/{len(images_to_process)}] "
                    f"{info.path.name} \u2192 {output_path.name}"
                )
            except (DngProcessingError, AvifEncodingError, MetadataError) as e:
                console.print(
                    f"[red]\u2717[/] [{i}/{len(images_to_process)}] "
                    f"{info.path.name}: {e}"
                )

        console.print(f"\n[bold green]Done![/] Processed {len(images_to_process)} image(s).")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/]")
        sys.exit(130)
    except ValidationError as e:
        console.print(f"\n[red]Configuration error:[/] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
