#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = "==3.14.*"
# dependencies = [
#     "pyexiftool>=0.5.6",
#     "rich>=13.0.0",
# ]
# ///
#
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2025 Aryan Ameri
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
Create ISO 21496-1 gain map AVIF from DNG, SDR AVIF, and HDR AVIF files.

Combines SDR and HDR AVIF exports (from Photomator) into a single gain map AVIF
that displays correctly on both SDR and HDR monitors. Copies metadata from the
original DNG file and syncs timestamps.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import exiftool
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
    "FileSet",
    "GainMapError",
    "ValidationError",
    "CombineError",
    "MetadataError",
    "find_file_sets",
    "create_gainmap_avif",
    "transfer_metadata",
    "sync_timestamp",
    "verify_output",
    "main",
]

__version__: Final[str] = "1.0.0"

# Path to avifgainmaputil built from source
SCRIPT_DIR: Final[Path] = Path(__file__).resolve().parent
AVIFGAINMAPUTIL: Final[Path] = SCRIPT_DIR / "avif_build" / "bin" / "avifgainmaputil"

# Console for rich output
console = Console()


# =============================================================================
# Exceptions
# =============================================================================


class GainMapError(Exception):
    """Base exception for gain map creation errors."""

    pass


class ValidationError(GainMapError):
    """Input validation failed (missing files or wrong format)."""

    pass


class CombineError(GainMapError):
    """Gain map combination via avifgainmaputil failed."""

    pass


class MetadataError(GainMapError):
    """Metadata transfer via exiftool failed."""

    pass


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True, slots=True)
class FileSet:
    """A triplet of DNG, SDR AVIF, and HDR AVIF files for processing."""

    dng_path: Path
    sdr_avif_path: Path
    hdr_avif_path: Path
    output_path: Path

    @property
    def basename(self) -> str:
        """Get the base filename without extension."""
        return self.dng_path.stem


@dataclass(frozen=True, slots=True)
class ProcessingConfig:
    """Configuration for gain map AVIF creation."""

    quality: int = 100
    gainmap_quality: int = 90
    speed: int = 3
    depth: int = 10
    dry_run: bool = False
    verbose: bool = False


# =============================================================================
# File Discovery
# =============================================================================


def find_file_sets(
    directory: Path,
    output_dir: Path | None = None,
) -> list[FileSet]:
    """
    Find DNG/SDR-AVIF/HDR-AVIF triplets by matching basenames.

    Expected naming convention (from Photomator):
    - IMG_XXXX.DNG      (source metadata)
    - IMG_XXXX.avif     (SDR: 8-bit sRGB)
    - IMG_XXXX 1.avif   (HDR: 10-bit PQ Rec.2100)

    Returns list of FileSet objects for processing.
    """
    file_sets: list[FileSet] = []
    out_dir = output_dir or directory

    # Find all DNG files (case-insensitive)
    dng_files = list(directory.glob("*.DNG")) + list(directory.glob("*.dng"))

    for dng_path in sorted(set(dng_files)):
        basename = dng_path.stem

        # Expected SDR and HDR AVIF paths
        sdr_avif_path = directory / f"{basename}.avif"
        hdr_avif_path = directory / f"{basename} 1.avif"
        output_path = out_dir / f"{basename}_hdr.avif"

        # Only include if both AVIF files exist
        if sdr_avif_path.exists() and hdr_avif_path.exists():
            file_sets.append(
                FileSet(
                    dng_path=dng_path,
                    sdr_avif_path=sdr_avif_path,
                    hdr_avif_path=hdr_avif_path,
                    output_path=output_path,
                )
            )

    return file_sets


# =============================================================================
# Core Processing Functions
# =============================================================================


def create_gainmap_avif(
    sdr_avif: Path,
    hdr_avif: Path,
    output_avif: Path,
    config: ProcessingConfig,
) -> bool:
    """
    Combine SDR base + HDR alternate into ISO 21496-1 gain map AVIF.

    Uses avifgainmaputil combine with archival quality settings.
    Returns True on success, raises CombineError on failure.
    """
    if config.dry_run:
        return True

    cmd = [
        str(AVIFGAINMAPUTIL),
        "combine",
        str(sdr_avif),
        str(hdr_avif),
        str(output_avif),
        "-q",
        str(config.quality),
        "--qgain-map",
        str(config.gainmap_quality),
        "--speed",
        str(config.speed),
        "--depth",
        str(config.depth),
        "--yuv",
        "444",
        "--depth-gain-map",
        "10",
        "--yuv-gain-map",
        "444",
        "--ignore-profile",  # Ignore ICC profiles, use CICP values instead
        # Set proper CICP color metadata for base (SDR) and alternate (HDR) images
        # Base: BT.709 primaries (1), sRGB transfer (13), BT.709 matrix (1)
        "--cicp-base",
        "1/13/1",
        # Alternate: BT.2020 primaries (9), PQ transfer (16), BT.2020 matrix (9)
        "--cicp-alternate",
        "9/16/9",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown error"
            raise CombineError(
                f"avifgainmaputil combine failed: {error_msg.strip()}"
            )

        return True

    except FileNotFoundError as e:
        raise CombineError(f"avifgainmaputil not found at {AVIFGAINMAPUTIL}") from e
    except subprocess.SubprocessError as e:
        raise CombineError(f"Failed to run avifgainmaputil: {e}") from e


def transfer_metadata(
    dng_path: Path,
    avif_path: Path,
    dry_run: bool = False,
) -> bool:
    """
    Copy all metadata from DNG to output AVIF using exiftool.

    Transfers EXIF, GPS, XMP, and MakerNotes while preserving
    AVIF color profile metadata (CICP).
    """
    if dry_run:
        return True

    try:
        with exiftool.ExifToolHelper() as et:
            # Copy all metadata from DNG to AVIF
            # Exclude color-related tags to preserve gain map CICP values
            et.execute(
                "-tagsFromFile",
                str(dng_path),
                "-all:all",
                "-overwrite_original",
                # Exclude CICP/color tags that avifgainmaputil set
                "-ColorPrimaries=",
                "-TransferCharacteristics=",
                "-MatrixCoefficients=",
                "-ColorSpace=",
                str(avif_path),
            )

        return True

    except Exception as e:
        raise MetadataError(f"Failed to transfer metadata: {e}") from e


def sync_timestamp(
    source: Path,
    target: Path,
    dry_run: bool = False,
) -> bool:
    """
    Copy file modification timestamp from source to target.

    Uses os.utime() to set atime/mtime matching the source file.
    """
    if dry_run:
        return True

    try:
        source_stat = source.stat()
        os.utime(target, (source_stat.st_atime, source_stat.st_mtime))
        return True
    except OSError as e:
        # Non-fatal, just log and continue
        console.print(f"[yellow]Warning: Could not sync timestamp: {e}[/yellow]")
        return False


def verify_output(avif_path: Path, verbose: bool = False) -> bool:
    """
    Verify the output AVIF is a valid ISO 21496-1 gain map AVIF.

    Checks via avifgainmaputil printmetadata that gain map is present.
    """
    if not avif_path.exists():
        return False

    try:
        result = subprocess.run(
            [str(AVIFGAINMAPUTIL), "printmetadata", str(avif_path)],
            capture_output=True,
            text=True,
            check=False,
        )

        # Check if output indicates gain map is present
        # The tool outputs "Gain Map Min:" when a gain map exists
        has_gainmap = "Gain Map Min" in result.stdout or "Base headroom" in result.stdout

        if verbose and result.stdout:
            console.print(f"[dim]{result.stdout.strip()}[/dim]")

        return has_gainmap

    except (FileNotFoundError, subprocess.SubprocessError):
        return False


# =============================================================================
# Processing Pipeline
# =============================================================================


def process_file_set(
    file_set: FileSet,
    config: ProcessingConfig,
) -> tuple[FileSet, bool, str]:
    """
    Process a single file set: combine, transfer metadata, sync timestamp.

    Returns (file_set, success, message) tuple.
    """
    try:
        # Phase 1: Validate inputs
        if not file_set.dng_path.exists():
            raise ValidationError(f"DNG not found: {file_set.dng_path}")
        if not file_set.sdr_avif_path.exists():
            raise ValidationError(f"SDR AVIF not found: {file_set.sdr_avif_path}")
        if not file_set.hdr_avif_path.exists():
            raise ValidationError(f"HDR AVIF not found: {file_set.hdr_avif_path}")

        # Phase 2: Combine into gain map AVIF
        create_gainmap_avif(
            file_set.sdr_avif_path,
            file_set.hdr_avif_path,
            file_set.output_path,
            config,
        )

        if not config.dry_run:
            # Phase 3: Transfer metadata from DNG
            transfer_metadata(
                file_set.dng_path,
                file_set.output_path,
                config.dry_run,
            )

            # Phase 4: Sync timestamp from DNG
            sync_timestamp(
                file_set.dng_path,
                file_set.output_path,
                config.dry_run,
            )

            # Verify output
            if not verify_output(file_set.output_path, config.verbose):
                return (file_set, False, "Output verification failed")

        return (file_set, True, "OK")

    except GainMapError as e:
        return (file_set, False, str(e))
    except Exception as e:
        return (file_set, False, f"Unexpected error: {e}")


def process_all(
    file_sets: list[FileSet],
    config: ProcessingConfig,
    jobs: int,
) -> tuple[int, int]:
    """
    Process all file sets with parallel execution and progress display.

    Returns (success_count, failure_count) tuple.
    """
    success_count = 0
    failure_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "[cyan]Creating gain map AVIFs...",
            total=len(file_sets),
        )

        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(process_file_set, fs, config): fs
                for fs in file_sets
            }

            for future in as_completed(futures):
                file_set, success, message = future.result()

                if success:
                    success_count += 1
                    if config.verbose:
                        status = "[dim][DRY RUN][/dim] " if config.dry_run else ""
                        console.print(
                            f"  {status}[green]✓[/green] {file_set.basename}"
                        )
                else:
                    failure_count += 1
                    console.print(
                        f"  [red]✗[/red] {file_set.basename}: {message}"
                    )

                progress.advance(task)

    return success_count, failure_count


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create ISO 21496-1 gain map AVIF from DNG, SDR AVIF, and HDR AVIF files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Combines SDR and HDR AVIF exports (from Photomator) into a single gain map AVIF
that displays correctly on both SDR and HDR monitors.

Expected input files:
  IMG_XXXX.DNG      - Original DNG with metadata
  IMG_XXXX.avif     - SDR AVIF (8-bit sRGB)
  IMG_XXXX 1.avif   - HDR AVIF (10-bit PQ Rec.2100)

Output:
  IMG_XXXX_hdr.avif - ISO 21496-1 gain map AVIF

Examples:
  %(prog)s                          Process current directory (archival quality)
  %(prog)s /path/to/images          Process specified directory
  %(prog)s -n -v                    Verbose dry run
  %(prog)s -s 6                     Faster encoding (lower quality)
        """,
    )

    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory containing DNG/SDR/HDR AVIF files (default: current directory)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        metavar="DIR",
        help="Output directory (default: same as input)",
    )
    parser.add_argument(
        "-q",
        "--quality",
        type=int,
        default=100,
        metavar="Q",
        help="AVIF quality for base image (0-100, default: 100)",
    )
    parser.add_argument(
        "--qgain-map",
        type=int,
        default=90,
        metavar="Q",
        help="Quality for gain map (0-100, default: 90)",
    )
    parser.add_argument(
        "-s",
        "--speed",
        type=int,
        default=3,
        metavar="S",
        help="Encoder speed (0-10, 0=slowest/best, default: 3)",
    )
    parser.add_argument(
        "-d",
        "--depth",
        type=int,
        default=10,
        choices=[8, 10, 12],
        metavar="D",
        help="Output bit depth (8, 10, 12, default: 10)",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Preview without making changes",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed output",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        metavar="N",
        help="Parallel processing jobs (default: CPU count)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()

    # Validate directory
    directory = Path(args.directory).resolve()
    if not directory.is_dir():
        console.print(f"[red]Error:[/red] '{directory}' is not a directory")
        sys.exit(1)

    # Validate output directory
    output_dir = args.output.resolve() if args.output else None
    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Check for required tools
    if not AVIFGAINMAPUTIL.exists():
        console.print(
            f"[red]Error:[/red] avifgainmaputil not found at {AVIFGAINMAPUTIL}"
        )
        console.print("Build it with: ./avif_tools.py")
        sys.exit(1)

    if not shutil.which("exiftool"):
        console.print("[red]Error:[/red] exiftool not found in PATH")
        console.print("Install with: sudo dnf install perl-Image-ExifTool")
        sys.exit(1)

    # Find file sets
    file_sets = find_file_sets(directory, output_dir)

    if not file_sets:
        console.print(f"[yellow]No DNG/SDR-AVIF/HDR-AVIF triplets found in {directory}[/yellow]")
        console.print("\nExpected files:")
        console.print("  IMG_XXXX.DNG")
        console.print("  IMG_XXXX.avif     (SDR)")
        console.print("  IMG_XXXX 1.avif   (HDR)")
        sys.exit(0)

    # Create config
    config = ProcessingConfig(
        quality=args.quality,
        gainmap_quality=args.qgain_map,
        speed=args.speed,
        depth=args.depth,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    # Determine job count
    jobs = args.jobs or os.cpu_count() or 1

    # Print header
    console.print()
    console.print(f"[bold]Create Gain Map AVIF v{__version__}[/bold]")
    console.print(f"Directory: {directory}")
    console.print(f"Found: {len(file_sets)} file set(s)")
    console.print(f"Quality: {config.quality} (gain map: {config.gainmap_quality})")
    console.print(f"Speed: {config.speed}, Depth: {config.depth}-bit")
    console.print(f"Jobs: {jobs}")
    if args.dry_run:
        console.print("[yellow][DRY RUN MODE][/yellow]")
    console.print()

    # Process all file sets
    success_count, failure_count = process_all(file_sets, config, jobs)

    # Print summary
    console.print()
    if failure_count == 0:
        console.print(
            f"[green]Complete:[/green] {success_count} file(s) processed successfully"
        )
    else:
        console.print(
            f"[yellow]Complete:[/yellow] {success_count} succeeded, "
            f"[red]{failure_count} failed[/red]"
        )

    sys.exit(0 if failure_count == 0 else 1)


if __name__ == "__main__":
    main()
