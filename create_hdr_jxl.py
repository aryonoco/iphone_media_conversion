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
Create Rec.2100 PQ HDR JXL from Lightroom-exported JXL and original ProRAW DNG.

This script takes a Lightroom-exported JXL (SDR with beautiful color grading) and
the original ProRAW DNG (with full HDR data), producing a direct Rec.2100 PQ HDR JXL.

Key insight: Preserve Lightroom's color grading (RGB ratios) while using the DNG's
full dynamic range for luminance. The result can be both brighter in highlights
and darker in shadows compared to SDR.

Why no gain map? iOS doesn't support JXL jhgm gain maps. This script creates a
native HDR image that iOS can display directly.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
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

__all__: Final[list[str]] = [
    "create_hdr_jxl",
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
# External Tool Wrappers
# =============================================================================


def decode_jxl_to_linear(jxl_path: Path, output_ppm: Path) -> None:
    """Decode JXL to 16-bit linear Rec.2020 PPM using djxl."""
    cmd = [
        str(DJXL_PATH),
        str(jxl_path),
        str(output_ppm),
        "--color_space=RGB_D65_202_Rel_Lin",
        "--bits_per_sample=16",
    ]
    result = run_command(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"djxl failed: {result.stderr}")


def decode_dng_to_linear(dng_path: Path, output_ppm: Path) -> None:
    """Decode DNG to 16-bit linear Rec.2020 PPM with HDR highlights preserved."""
    cmd = [
        str(DCRAW_EMU_PATH),
        "-dngsdk",  # Required for JXL-compressed ProRAW
        "-4",  # 16-bit linear
        "-H", "1",  # Preserve highlights (unclip)
        "-o", "8",  # Rec.2020 output
        "-q", "3",  # AHD demosaicing
        "-Z", str(output_ppm),
        str(dng_path),
    ]
    result = run_command(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"dcraw_emu_dng failed: {result.stderr}")


def encode_hdr_jxl(
    ppm_path: Path,
    output_jxl: Path,
    quality: float = 0.5,
) -> None:
    """Encode PQ PPM as Rec.2100 PQ JXL."""
    cmd = [
        str(CJXL_PATH),
        str(ppm_path),
        str(output_jxl),
        "-x", "color_space=Rec2100PQ",  # Tell cjxl input is PQ
        "-d", str(quality),  # Visually lossless
        "-e", "9",  # High effort
    ]
    result = run_command(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"cjxl failed: {result.stderr}")


# =============================================================================
# PPM I/O
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


def write_pq_ppm(path: Path, pq_data: NDArray[np.float32]) -> None:
    """Write PQ-encoded data as 16-bit PPM."""
    height, width = pq_data.shape[:2]

    # Scale PQ [0, 1] to 16-bit [0, 65535]
    data_16bit = (np.clip(pq_data, 0, 1) * 65535).astype(np.uint16)

    with open(path, "wb") as f:
        f.write(f"P6\n{width} {height}\n65535\n".encode("ascii"))
        # PPM uses big-endian for 16-bit values
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
    """
    hdr_h, hdr_w = hdr_linear.shape[:2]
    sdr_h, sdr_w = sdr_shape

    # If dimensions already match, no rotation needed
    if (hdr_h, hdr_w) == (sdr_h, sdr_w):
        return hdr_linear

    # If dimensions are swapped, we need to rotate
    if (hdr_h, hdr_w) == (sdr_w, sdr_h):
        dng_orient = get_orientation(dng_path)

        if dng_orient == 6:
            return np.rot90(hdr_linear, k=-1)
        elif dng_orient == 8:
            return np.rot90(hdr_linear, k=1)
        else:
            return np.rot90(hdr_linear, k=-1)

    raise ValueError(
        f"Cannot match dimensions: HDR ({hdr_h}, {hdr_w}) to SDR ({sdr_h}, {sdr_w})"
    )


# =============================================================================
# HDR Processing Core
# =============================================================================


def align_hdr_to_sdr(
    sdr_linear: NDArray[np.float32],
    hdr_linear: NDArray[np.float32],
    low_percentile: float = 20.0,
    high_percentile: float = 80.0,
) -> NDArray[np.float32]:
    """Scale HDR so its mid-tone luminance matches SDR's mid-tones.

    Uses percentile band (20-80%) for robust alignment across scene types.
    """
    sdr_lum = np.dot(sdr_linear, Y_COEFFS).flatten()
    hdr_lum = np.dot(hdr_linear, Y_COEFFS).flatten()

    # Use percentile band for robust alignment
    sdr_low, sdr_high = np.percentile(sdr_lum, [low_percentile, high_percentile])
    mask = (sdr_lum >= sdr_low) & (sdr_lum <= sdr_high)

    if mask.sum() == 0:
        # Fallback to median if no pixels in range
        scale = float(np.median(sdr_lum) / np.maximum(np.median(hdr_lum), 1e-10))
    else:
        sdr_mean = float(np.mean(sdr_lum[mask]))
        hdr_mean = float(np.mean(hdr_lum[mask]))
        scale = sdr_mean / np.maximum(hdr_mean, 1e-10)

    return (hdr_linear * scale).astype(np.float32)


def create_hdr_from_sdr_and_dng(
    sdr_linear: NDArray[np.float32],
    hdr_aligned: NDArray[np.float32],
    shadow_threshold: float = 0.01,
) -> NDArray[np.float32]:
    """Create HDR image preserving Lightroom's color grading.

    Strategy:
    - Keep SDR's RGB ratios (chromaticity/color grading)
    - Apply DNG's luminance ratio (can brighten OR darken)
    - Smooth ratio in deep shadows to avoid noise amplification
    - Result: Lightroom's colors + DNG's full dynamic range
    """
    sdr_lum = np.dot(sdr_linear, Y_COEFFS)
    hdr_lum = np.dot(hdr_aligned, Y_COEFFS)

    # Full HDR ratio - can be > 1 (brighter) or < 1 (darker)
    ratio = hdr_lum / np.maximum(sdr_lum, 1e-10)

    # Shadow noise protection: blend ratio towards 1.0 in deep shadows
    shadow_weight = np.clip(sdr_lum / shadow_threshold, 0, 1)
    ratio = shadow_weight * ratio + (1 - shadow_weight) * 1.0

    # Apply ratio to SDR colors
    hdr_output = sdr_linear * ratio[:, :, np.newaxis]

    return np.clip(hdr_output, 0, None).astype(np.float32)


def linear_to_pq(
    linear_rgb: NDArray[np.float32],
    sdr_white_nits: float,
) -> NDArray[np.float32]:
    """Convert linear light to PQ transfer function (SMPTE ST 2084).

    Args:
        linear_rgb: Linear RGB values where 1.0 = SDR white
        sdr_white_nits: SDR reference white in nits (default 203 per ITU-R BT.2408)
    """
    # Convert linear (where 1.0 = SDR white) to absolute nits
    nits = linear_rgb * sdr_white_nits

    # Normalize to PQ range (10000 nits = 1.0)
    Y = np.clip(nits / 10000.0, 0, 1)

    # PQ EOTF constants (SMPTE ST 2084)
    m1 = 2610.0 / 16384.0
    m2 = 2523.0 / 4096.0 * 128.0
    c1 = 3424.0 / 4096.0
    c2 = 2413.0 / 4096.0 * 32.0
    c3 = 2392.0 / 4096.0 * 32.0

    Ym1 = np.power(Y, m1)
    pq = np.power((c1 + c2 * Ym1) / (1.0 + c3 * Ym1), m2)

    return pq.astype(np.float32)


# =============================================================================
# Main Pipeline
# =============================================================================


def create_hdr_jxl(
    lightroom_jxl: Path,
    dng_path: Path,
    output_jxl: Path,
    sdr_white_nits: float = 203.0,
    quality: float = 0.5,
    verbose: bool = False,
) -> None:
    """Create Rec.2100 PQ HDR JXL from Lightroom SDR + DNG.

    Args:
        lightroom_jxl: Path to Lightroom-exported SDR JXL
        dng_path: Path to original ProRAW DNG
        output_jxl: Output path for HDR JXL
        sdr_white_nits: SDR reference white in nits (default 203 per ITU-R BT.2408)
        quality: JXL quality (distance), 0.5 = high quality HDR, 1.0 = visually lossless
        verbose: Print detailed progress
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # 1. Decode sources to 16-bit linear Rec.2020
        if verbose:
            console.print("[dim]Decoding SDR JXL to linear Rec.2020...[/dim]")
        sdr_ppm = tmp / "sdr.ppm"
        decode_jxl_to_linear(lightroom_jxl, sdr_ppm)

        if verbose:
            console.print("[dim]Decoding DNG to linear Rec.2020...[/dim]")
        hdr_ppm = tmp / "hdr.ppm"
        decode_dng_to_linear(dng_path, hdr_ppm)

        # 2. Read pixel data as float for processing
        if verbose:
            console.print("[dim]Reading pixel data...[/dim]")
        sdr_linear = read_ppm_as_float(sdr_ppm)
        hdr_linear = read_ppm_as_float(hdr_ppm)

        # 3. Apply BaselineExposure to HDR
        baseline_ev = get_baseline_exposure(dng_path)
        if verbose:
            console.print(f"[dim]BaselineExposure: {baseline_ev:.2f} EV[/dim]")
        hdr_linear = (hdr_linear * (2.0 ** baseline_ev)).astype(np.float32)

        # 4. Handle rotation if needed
        if verbose:
            console.print(
                f"[dim]SDR shape: {sdr_linear.shape}, HDR shape: {hdr_linear.shape}[/dim]"
            )
        hdr_linear = apply_rotation_to_match(
            hdr_linear, sdr_linear.shape[:2], dng_path
        )

        # 5. Align HDR mid-tones to SDR
        if verbose:
            console.print("[dim]Aligning HDR to SDR mid-tones...[/dim]")
        hdr_aligned = align_hdr_to_sdr(sdr_linear, hdr_linear)

        # 6. Create HDR by applying luminance ratio
        if verbose:
            console.print("[dim]Creating HDR from SDR colors + DNG luminance...[/dim]")
        hdr_output = create_hdr_from_sdr_and_dng(sdr_linear, hdr_aligned)

        # 7. Convert to PQ with configurable SDR white point
        if verbose:
            console.print(f"[dim]Converting to PQ (SDR white = {sdr_white_nits} nits)...[/dim]")
        hdr_pq = linear_to_pq(hdr_output, sdr_white_nits)

        # 8. Write as 16-bit PPM
        if verbose:
            console.print("[dim]Writing PQ PPM...[/dim]")
        hdr_pq_ppm = tmp / "hdr_pq.ppm"
        write_pq_ppm(hdr_pq_ppm, hdr_pq)

        # 9. Encode as Rec.2100 PQ JXL
        if verbose:
            console.print("[dim]Encoding as Rec.2100 PQ JXL...[/dim]")
        encode_hdr_jxl(hdr_pq_ppm, output_jxl, quality)


# =============================================================================
# Batch Processing
# =============================================================================


@dataclass(frozen=True, slots=True)
class FileMatch:
    """Matched JXL and DNG file pair."""
    jxl_path: Path
    dng_path: Path
    output_path: Path


def find_matching_files(directory: Path, output_suffix: str = "_hdr") -> list[FileMatch]:
    """Find matching JXL and DNG files in a directory.

    Looks for patterns like:
    - IMG_1234.jxl + IMG_1234.DNG
    - photo.jxl + photo.DNG
    """
    matches: list[FileMatch] = []

    for jxl_file in directory.glob("*.jxl"):
        # Skip files that already have HDR suffix
        if jxl_file.stem.endswith(output_suffix):
            continue

        # Look for matching DNG
        dng_file = jxl_file.with_suffix(".DNG")
        if not dng_file.exists():
            dng_file = jxl_file.with_suffix(".dng")

        if dng_file.exists():
            output_file = jxl_file.with_stem(jxl_file.stem + output_suffix)
            matches.append(FileMatch(jxl_file, dng_file, output_file))

    return sorted(matches, key=lambda m: m.jxl_path.name)


def process_batch(
    directory: Path,
    sdr_white_nits: float = 203.0,
    quality: float = 0.5,
    verbose: bool = False,
) -> None:
    """Process all matching JXL+DNG pairs in a directory."""
    matches = find_matching_files(directory)

    if not matches:
        console.print("[yellow]No matching JXL+DNG pairs found.[/yellow]")
        return

    console.print(f"Found [bold]{len(matches)}[/bold] file pairs to process.")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing...", total=len(matches))

        for match in matches:
            progress.update(task, description=f"[cyan]{match.jxl_path.name}[/cyan]")

            try:
                create_hdr_jxl(
                    match.jxl_path,
                    match.dng_path,
                    match.output_path,
                    sdr_white_nits=sdr_white_nits,
                    quality=quality,
                    verbose=verbose,
                )
                if verbose:
                    console.print(f"  [green]Created {match.output_path.name}[/green]")
            except Exception as e:
                console.print(f"  [red]Error: {e}[/red]")

            progress.advance(task)


# =============================================================================
# CLI
# =============================================================================


def validate_tools() -> None:
    """Validate required external tools exist."""
    missing = []

    if not DJXL_PATH.exists():
        missing.append(f"djxl not found at {DJXL_PATH}")
    if not CJXL_PATH.exists():
        missing.append(f"cjxl not found at {CJXL_PATH}")
    if not DCRAW_EMU_PATH.exists():
        missing.append(f"dcraw_emu_dng not found at {DCRAW_EMU_PATH}")

    # Check exiftool
    try:
        subprocess.run(["exiftool", "-ver"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        missing.append("exiftool not found in PATH")

    if missing:
        console.print("[red]Missing required tools:[/red]")
        for m in missing:
            console.print(f"  - {m}")
        console.print("\n[dim]Run libraw_dng.py to build the required tools.[/dim]")
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create Rec.2100 PQ HDR JXL from Lightroom JXL + ProRAW DNG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file mode
  %(prog)s --source photo.jxl --dng photo.DNG --output photo_hdr.jxl

  # Batch mode
  %(prog)s --dir /path/to/photos

  # Custom SDR white point
  %(prog)s --dir /path/to/photos --sdr-white-nits 100
""",
    )

    # Single file mode
    parser.add_argument(
        "--source",
        type=Path,
        help="Path to Lightroom-exported SDR JXL",
    )
    parser.add_argument(
        "--dng",
        type=Path,
        help="Path to original ProRAW DNG",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for HDR JXL",
    )

    # Batch mode
    parser.add_argument(
        "--dir",
        type=Path,
        help="Directory for batch processing (finds matching JXL+DNG pairs)",
    )

    # Options
    parser.add_argument(
        "--sdr-white-nits",
        type=float,
        default=203.0,
        help="SDR reference white in nits (default: 203 per ITU-R BT.2408)",
    )
    parser.add_argument(
        "-q", "--quality",
        type=float,
        default=0.5,
        help="JXL quality (distance), 0.5 = high quality HDR (default: 0.5)",
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

    # Validate tools
    validate_tools()

    # Determine mode
    if args.dir:
        # Batch mode
        if not args.dir.is_dir():
            console.print(f"[red]Not a directory: {args.dir}[/red]")
            sys.exit(1)

        process_batch(
            args.dir,
            sdr_white_nits=args.sdr_white_nits,
            quality=args.quality,
            verbose=args.verbose,
        )

    elif args.source and args.dng and args.output:
        # Single file mode
        if not args.source.exists():
            console.print(f"[red]Source file not found: {args.source}[/red]")
            sys.exit(1)
        if not args.dng.exists():
            console.print(f"[red]DNG file not found: {args.dng}[/red]")
            sys.exit(1)

        console.print(f"Processing [cyan]{args.source.name}[/cyan]...")
        create_hdr_jxl(
            args.source,
            args.dng,
            args.output,
            sdr_white_nits=args.sdr_white_nits,
            quality=args.quality,
            verbose=args.verbose,
        )
        console.print(f"[green]Created {args.output}[/green]")

    else:
        parser.print_help()
        console.print("\n[yellow]Specify either --dir for batch mode or --source/--dng/--output for single file.[/yellow]")
        sys.exit(1)


if __name__ == "__main__":
    main()
