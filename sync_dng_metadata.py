#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pyexiftool>=0.5.6",
# ]
# ///
"""
Sync metadata from DNG files to converted image files (AVIF, JXL).

Transfers all EXIF, GPS, XMP, and MakerNotes metadata from original DNG files
to their corresponding converted images. Also syncs file modification timestamps.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import exiftool

# Supported target formats
FORMAT_EXTENSIONS = {
    "avif": [".avif"],
    "jxl": [".jxl"],
    "png": [".png"],
    "heic": [".heic"],
    "all": [".avif", ".jxl", ".png", ".heic"],
}


def find_pairs(directory: Path, extensions: list[str]) -> list[tuple[Path, Path]]:
    """Find DNG/target pairs by matching basenames."""
    pairs = []

    for dng_file in directory.glob("*.DNG"):
        for ext in extensions:
            target_file = dng_file.with_suffix(ext)
            if target_file.exists():
                pairs.append((dng_file, target_file))

    # Also check lowercase .dng
    for dng_file in directory.glob("*.dng"):
        for ext in extensions:
            target_file = dng_file.with_suffix(ext)
            if target_file.exists():
                pairs.append((dng_file, target_file))

    return pairs


def sync_metadata(
    dng_path: Path,
    target_path: Path,
    dry_run: bool = False,
    verbose: bool = False,
) -> bool:
    """
    Copy all metadata from DNG to target file using exiftool.

    Returns True on success, False on failure.
    """
    if verbose:
        print(f"  Source: {dng_path.name}")
        print(f"  Target: {target_path.name}")

    if dry_run:
        print(f"  [DRY RUN] Would copy metadata")
        return True

    try:
        with exiftool.ExifToolHelper() as et:
            # Copy all metadata from DNG to target
            # Using execute to run raw exiftool command for full control
            et.execute(
                "-tagsFromFile", str(dng_path),
                "-all:all",
                "-overwrite_original",
                str(target_path)
            )

        if verbose:
            print(f"  Metadata copied successfully")

        return True

    except Exception as e:
        print(f"  ERROR: Failed to copy metadata: {e}", file=sys.stderr)
        return False


def sync_timestamp(
    dng_path: Path,
    target_path: Path,
    dry_run: bool = False,
    verbose: bool = False,
) -> bool:
    """
    Copy file modification timestamp from DNG to target file.

    Returns True on success, False on failure.
    """
    try:
        dng_stat = dng_path.stat()

        if dry_run:
            if verbose:
                print(f"  [DRY RUN] Would set mtime to {dng_stat.st_mtime}")
            return True

        # Set both access time and modification time
        os.utime(target_path, (dng_stat.st_atime, dng_stat.st_mtime))

        if verbose:
            print(f"  File timestamp synced")

        return True

    except Exception as e:
        print(f"  ERROR: Failed to sync timestamp: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Sync metadata from DNG files to converted image files (AVIF, JXL).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    Sync all formats in current directory
  %(prog)s /path/to/images    Sync pairs in specified directory
  %(prog)s --format jxl       Sync only JXL files
  %(prog)s --format avif      Sync only AVIF files
  %(prog)s --dry-run          Preview without making changes
  %(prog)s -v                 Show detailed output
        """
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory containing DNG and target files (default: current directory)"
    )
    parser.add_argument(
        "-f", "--format",
        choices=["avif", "jxl", "png", "heic", "all"],
        default="all",
        help="Target format(s) to sync: avif, jxl, png, heic, or all (default: all)"
    )
    parser.add_argument(
        "-n", "--dry-run",
        action="store_true",
        help="Preview changes without modifying files"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed output"
    )

    args = parser.parse_args()

    directory = Path(args.directory).resolve()

    if not directory.is_dir():
        print(f"Error: '{directory}' is not a directory", file=sys.stderr)
        sys.exit(1)

    # Check for exiftool
    if not shutil.which("exiftool"):
        print("Error: exiftool not found in PATH", file=sys.stderr)
        print("Install with: sudo dnf install perl-Image-ExifTool", file=sys.stderr)
        sys.exit(1)

    # Get target extensions based on format choice
    extensions = FORMAT_EXTENSIONS[args.format]
    format_desc = args.format.upper() if args.format != "all" else "AVIF/JXL/PNG/HEIC"

    # Find pairs
    pairs = find_pairs(directory, extensions)

    if not pairs:
        print(f"No DNG/{format_desc} pairs found in {directory}")
        sys.exit(0)

    print(f"Found {len(pairs)} DNG/{format_desc} pair(s) in {directory}")
    if args.dry_run:
        print("[DRY RUN MODE - no files will be modified]")
    print()

    success_count = 0
    fail_count = 0

    for dng_path, target_path in pairs:
        print(f"Processing: {target_path.name}")

        # Sync metadata
        meta_ok = sync_metadata(
            dng_path, target_path,
            dry_run=args.dry_run,
            verbose=args.verbose
        )

        # Sync timestamp
        time_ok = sync_timestamp(
            dng_path, target_path,
            dry_run=args.dry_run,
            verbose=args.verbose
        )

        if meta_ok and time_ok:
            success_count += 1
        else:
            fail_count += 1

        if args.verbose:
            print()

    # Summary
    print()
    print(f"Complete: {success_count} succeeded, {fail_count} failed")

    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
