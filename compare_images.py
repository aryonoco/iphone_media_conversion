#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=1.26", "scipy>=1.11"]
# ///

"""Compare pixel values between script JXL and Apple PNG outputs.

Test infrastructure for incremental improvements to export_photos.py.
Compares luminance statistics between script output and Apple reference.

IMPORTANT: Apple PNG uses PQ (ST 2084) transfer function, script JXL uses sRGB.
Direct pixel comparison is meaningless - must convert to LINEAR first!
"""

import subprocess
import numpy as np
import os
import argparse
from pathlib import Path

# Set library path for libjxl
os.environ["LD_LIBRARY_PATH"] = "/var/home/admin/projects/iphone_media_conversion/build/deps/lib64:/home/linuxbrew/.linuxbrew/lib"

DJXL = "/var/home/admin/projects/iphone_media_conversion/build/deps/bin/djxl"
JXLINFO = "/var/home/admin/projects/iphone_media_conversion/build/deps/bin/jxlinfo"

# SDR reference white in nits (for normalizing PQ to relative luminance)
SDR_WHITE_NITS = 203.0  # ITU-R BT.2408 reference


def extract_gainmap_from_jxl(jxl_path: Path) -> tuple[np.ndarray, dict] | None:
    """Extract gain map and metadata from JXL file.

    Returns:
        Tuple of (gain_map_image, metadata_dict) or None if no gain map
    """
    import struct
    import tempfile

    with open(jxl_path, 'rb') as f:
        data = f.read()

    # JXL container signature
    JXL_SIG = bytes([0x00, 0x00, 0x00, 0x0C, 0x4A, 0x58, 0x4C, 0x20, 0x0D, 0x0A, 0x87, 0x0A])
    if not data.startswith(JXL_SIG):
        print(f"  Not a JXL container: {jxl_path}")
        return None

    # Parse ISOBMFF boxes to find jhgm
    pos = 12  # Skip signature
    jhgm_data = None

    while pos < len(data) - 8:
        box_size = struct.unpack('>I', data[pos:pos+4])[0]
        box_type = data[pos+4:pos+8].decode('ascii', errors='replace')

        if box_size == 0:
            break
        if box_size == 1:  # Extended size
            box_size = struct.unpack('>Q', data[pos+8:pos+16])[0]

        if box_type == 'jhgm':
            jhgm_data = data[pos+8:pos+box_size]
            break

        pos += box_size

    if jhgm_data is None:
        print(f"  No jhgm box found in {jxl_path}")
        return None

    # Parse jhgm bundle (from lib/extras/gain_map.cc):
    # - jhgm_version: 1 byte (0x00)
    # - gain_map_metadata_size: 2 bytes (BE16)
    # - gain_map_metadata: N bytes (ISO 21496-1 binary)
    # - color_encoding_size: 1 byte
    # - color_encoding: N bytes (if present)
    # - alt_icc_size: 4 bytes (BE32)
    # - alt_icc: N bytes (if present)
    # - gain_map: remaining bytes (naked JXL codestream)

    pos = 0
    version = jhgm_data[pos]
    pos += 1

    metadata_size = struct.unpack('>H', jhgm_data[pos:pos+2])[0]  # 2 bytes BE16
    pos += 2

    metadata_bytes = jhgm_data[pos:pos+metadata_size]
    pos += metadata_size

    color_encoding_size = jhgm_data[pos]
    pos += 1 + color_encoding_size  # Skip color encoding

    alt_icc_size = struct.unpack('>I', jhgm_data[pos:pos+4])[0]
    pos += 4 + alt_icc_size  # Skip alt ICC

    gainmap_jxl = jhgm_data[pos:]

    # Parse ISO 21496-1 metadata
    # Format: version(4) + flags(1) + base_headroom(8) + alt_headroom(8) + channel_data(40)
    # Skip version info (4 bytes) and flags (1 byte)
    # Then base_hdr_headroom (8), alternate_hdr_headroom (8)
    # Then channel data: min (8), max (8), gamma (8), base_offset (8), alt_offset (8)
    meta_offset = 5 + 8 + 8  # Skip header, base headroom, alt headroom

    def read_fraction(data, pos):
        num = struct.unpack('>i', data[pos:pos+4])[0]
        denom = struct.unpack('>I', data[pos+4:pos+8])[0]
        return num / denom if denom != 0 else 0.0

    # ISO 21496-1 metadata layout after header:
    # offset 0: gain_map_min (8 bytes)
    # offset 8: gain_map_max (8 bytes)
    # offset 16: gain_map_gamma (8 bytes)
    # offset 24: base_offset (8 bytes)
    # offset 32: alternate_offset (8 bytes)
    gain_map_min = read_fraction(metadata_bytes, meta_offset)
    gain_map_max = read_fraction(metadata_bytes, meta_offset + 8)
    gamma = read_fraction(metadata_bytes, meta_offset + 16)
    base_offset = read_fraction(metadata_bytes, meta_offset + 24)

    metadata = {
        'gain_map_min': gain_map_min,
        'gain_map_max': gain_map_max,
        'gamma': gamma if gamma > 0 else 1.0,
        'offset': base_offset if base_offset > 0 else 1/64,  # Default to 1/64
    }

    # Decode gain map JXL codestream to PNG (grayscale doesn't work with PPM)
    with tempfile.NamedTemporaryFile(suffix='.jxl', delete=False) as f:
        f.write(gainmap_jxl)
        gainmap_jxl_path = Path(f.name)

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        gainmap_png_path = Path(f.name)

    result = subprocess.run([DJXL, str(gainmap_jxl_path), str(gainmap_png_path)],
                           capture_output=True, text=True, env=os.environ)

    os.unlink(gainmap_jxl_path)

    if result.returncode != 0:
        print(f"  Failed to decode gain map: {result.stderr}")
        return None

    # Read grayscale gain map using ImageMagick
    gainmap_img = read_grayscale_png(gainmap_png_path)
    os.unlink(gainmap_png_path)

    if gainmap_img is None:
        print(f"  Failed to read gain map PNG")
        return None

    return gainmap_img, metadata


def reconstruct_hdr(sdr_srgb: np.ndarray, gainmap: np.ndarray, metadata: dict) -> np.ndarray:
    """Reconstruct HDR from SDR base and gain map.

    Gain map was computed as:
        pixel_gain = (hdr_lum + offset) / (sdr_lum + offset)

    Reconstruction:
        (hdr_lum + offset) = (sdr_lum + offset) * pixel_gain
        hdr_lum = (sdr_lum + offset) * pixel_gain - offset

    For RGB, we scale each channel proportionally based on luminance ratio.

    Args:
        sdr_srgb: SDR image in sRGB encoding (0-1)
        gainmap: Gain map image (0-1, normalized)
        metadata: Gain map metadata with min/max/gamma/offset

    Returns:
        HDR image in linear light (can exceed 1.0)
    """
    # Convert SDR to linear
    sdr_linear = srgb_to_linear(sdr_srgb)

    # Resize gain map if needed (it might be lower resolution)
    sdr_h, sdr_w = sdr_linear.shape[:2]
    gm_h, gm_w = gainmap.shape[:2]
    if (gm_h, gm_w) != (sdr_h, sdr_w):
        # Use bilinear interpolation to upscale gain map
        from scipy import ndimage
        scale_y = sdr_h / gm_h
        scale_x = sdr_w / gm_w
        gainmap = ndimage.zoom(gainmap, (scale_y, scale_x), order=1)

    # Decode gain map values to log2 gain
    gain_map_min = metadata['gain_map_min']
    gain_map_max = metadata['gain_map_max']
    gamma = metadata.get('gamma', 1.0)
    offset = metadata.get('offset', 1/64)

    # Reverse gamma encoding if applied
    normalized = gainmap
    if gamma != 1.0:
        normalized = np.power(normalized, gamma)

    # Decode log2 gain
    log_gain = gain_map_min + normalized * (gain_map_max - gain_map_min)

    # Convert to linear gain
    pixel_gain = np.power(2.0, log_gain)

    # Compute SDR luminance (Rec.2020 coefficients)
    sdr_lum = 0.2627 * sdr_linear[:,:,0] + 0.6780 * sdr_linear[:,:,1] + 0.0593 * sdr_linear[:,:,2]

    # Reconstruct HDR luminance with offset:
    # (hdr_lum + offset) = (sdr_lum + offset) * pixel_gain
    hdr_lum = (sdr_lum + offset) * pixel_gain - offset
    hdr_lum = np.maximum(hdr_lum, 0.0)  # Clamp negative values

    # Compute scale factor to apply to RGB
    # Avoid division by zero
    scale = np.divide(hdr_lum, sdr_lum, out=np.ones_like(hdr_lum), where=sdr_lum > 1e-10)

    # Apply scale to each channel
    hdr_linear = sdr_linear * scale[:, :, np.newaxis]

    return hdr_linear


def srgb_to_linear(img: np.ndarray) -> np.ndarray:
    """Convert sRGB-encoded values to linear light.

    Applies the inverse sRGB EOTF (electro-optical transfer function).
    Input: sRGB values in range [0, 1]
    Output: Linear light values in range [0, 1]
    """
    # sRGB EOTF breakpoint
    threshold = 0.04045

    # Vectorized piecewise function
    linear = np.where(
        img <= threshold,
        img / 12.92,
        np.power((img + 0.055) / 1.055, 2.4)
    )
    return linear.astype(np.float32)


def pq_to_linear(img: np.ndarray, normalize_to_sdr: bool = True) -> np.ndarray:
    """Convert PQ (ST 2084) encoded values to linear light.

    Applies the inverse PQ EOTF.
    Input: PQ values in range [0, 1]
    Output: Linear light values (absolute nits or normalized to SDR white)

    Args:
        img: PQ-encoded image data
        normalize_to_sdr: If True, normalize output so SDR white (203 nits) = 1.0
                         If False, return absolute nits (0-10000)
    """
    # PQ (SMPTE ST 2084) constants
    m1 = 0.1593017578125  # 2610/16384
    m2 = 78.84375         # 2523/32 * 128
    c1 = 0.8359375        # 3424/4096
    c2 = 18.8515625       # 2413/128
    c3 = 18.6875          # 2392/128

    # Avoid division by zero
    img = np.clip(img, 1e-10, 1.0)

    # Inverse PQ EOTF
    Np = np.power(img, 1.0 / m2)
    numerator = np.maximum(Np - c1, 0.0)
    denominator = c2 - c3 * Np
    # Avoid division by zero
    denominator = np.maximum(denominator, 1e-10)

    linear = np.power(numerator / denominator, 1.0 / m1)

    # Scale to absolute luminance (0-10000 nits)
    linear_nits = linear * 10000.0

    if normalize_to_sdr:
        # Normalize so SDR white (203 nits) = 1.0
        # This makes comparison with SDR content meaningful
        return (linear_nits / SDR_WHITE_NITS).astype(np.float32)
    else:
        return linear_nits.astype(np.float32)


def decode_jxl_to_ppm(jxl_path: Path, ppm_path: Path) -> bool:
    """Decode JXL to PPM using djxl."""
    result = subprocess.run([DJXL, str(jxl_path), str(ppm_path)],
                          capture_output=True, text=True, env=os.environ)
    return result.returncode == 0


def read_ppm(path: Path) -> np.ndarray:
    """Read PPM file to numpy array (normalized 0-1)."""
    with open(path, 'rb') as f:
        magic = f.readline()
        line = f.readline()
        while line.startswith(b'#'):
            line = f.readline()
        width, height = map(int, line.decode().split())
        maxval = int(f.readline().decode().strip())
        data = f.read()

    if maxval == 65535:
        img = np.frombuffer(data, dtype='>u2').reshape(height, width, 3)
    else:
        img = np.frombuffer(data, dtype='u1').reshape(height, width, 3)

    return img.astype(np.float32) / maxval


def read_png(path: Path) -> np.ndarray | None:
    """Read PNG using ImageMagick convert."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.ppm', delete=False) as f:
        ppm_path = f.name

    # Use ImageMagick to convert PNG to PPM (handles HDR PNGs)
    result = subprocess.run(
        ['convert', str(path), '-depth', '16', f'ppm:{ppm_path}'],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Error converting {path}: {result.stderr}")
        return None

    img = read_ppm(Path(ppm_path))
    os.unlink(ppm_path)
    return img


def read_grayscale_png(path: Path) -> np.ndarray | None:
    """Read grayscale PNG using ImageMagick convert to PGM."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pgm', delete=False) as f:
        pgm_path = f.name

    # Convert to PGM (portable graymap)
    result = subprocess.run(
        ['convert', str(path), '-depth', '16', f'pgm:{pgm_path}'],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Error converting grayscale {path}: {result.stderr}")
        return None

    # Read PGM file
    with open(pgm_path, 'rb') as f:
        magic = f.readline()  # P5 for binary PGM
        line = f.readline()
        while line.startswith(b'#'):
            line = f.readline()
        width, height = map(int, line.decode().split())
        maxval = int(f.readline().decode().strip())
        data = f.read()

    os.unlink(pgm_path)

    if maxval == 65535:
        img = np.frombuffer(data, dtype='>u2').reshape(height, width)
    else:
        img = np.frombuffer(data, dtype='u1').reshape(height, width)

    return img.astype(np.float32) / maxval


def compute_luminance(img: np.ndarray) -> np.ndarray:
    """Compute luminance using Rec.2020 coefficients."""
    return 0.2627 * img[:,:,0] + 0.6780 * img[:,:,1] + 0.0593 * img[:,:,2]


def analyze_image(name: str, img: np.ndarray, verbose: bool = True) -> dict:
    """Compute statistics for an image."""
    lum = compute_luminance(img)

    stats = {
        "shape": img.shape,
        "range": (float(img.min()), float(img.max())),
        "mean": float(img.mean()),
        "lum_range": (float(lum.min()), float(lum.max())),
        "lum_mean": float(lum.mean()),
        "lum_median": float(np.median(lum)),
        "lum_percentiles": {p: float(np.percentile(lum, p)) for p in [1, 5, 25, 50, 75, 95, 99]},
    }

    if verbose:
        print(f"\n{name}:")
        print(f"  Shape: {stats['shape']}")
        print(f"  Range: [{stats['range'][0]:.6f}, {stats['range'][1]:.6f}]")
        print(f"  Mean: {stats['mean']:.6f}")
        print(f"  Luminance range: [{stats['lum_range'][0]:.6f}, {stats['lum_range'][1]:.6f}]")
        print(f"  Luminance mean: {stats['lum_mean']:.6f}")
        print(f"  Luminance percentiles:")
        for p, v in stats['lum_percentiles'].items():
            print(f"    {p}%: {v:.6f}")

    return stats


def compare_images(script_jxl: Path, apple_png: Path, name: str, verbose: bool = True,
                   mode: str = "hdr") -> dict | None:
    """Compare a single image pair and return comparison metrics.

    Args:
        script_jxl: Path to script-generated JXL file
        apple_png: Path to Apple reference PNG file
        name: Image name for display
        verbose: Print detailed statistics
        mode: Comparison mode:
              - "hdr": Reconstruct HDR from gain map and compare (recommended!)
              - "sdr": Compare SDR base (linearized) vs Apple
              - "raw": Compare raw encoded values (meaningless)
    """
    import tempfile

    if not script_jxl.exists():
        print(f"  Script JXL not found: {script_jxl}")
        return None

    if not apple_png.exists():
        print(f"  Apple PNG not found: {apple_png}")
        return None

    # Decode script JXL (SDR base)
    with tempfile.NamedTemporaryFile(suffix='.ppm', delete=False) as f:
        ppm_path = Path(f.name)

    if not decode_jxl_to_ppm(script_jxl, ppm_path):
        print(f"  Failed to decode {script_jxl}")
        return None

    script_sdr = read_ppm(ppm_path)
    os.unlink(ppm_path)

    # Read Apple PNG
    apple_img = read_png(apple_png)
    if apple_img is None:
        return None

    if mode == "hdr":
        # FULL HDR COMPARISON: Reconstruct HDR from gain map
        gainmap_result = extract_gainmap_from_jxl(script_jxl)
        if gainmap_result is None:
            print(f"  No gain map found, falling back to SDR comparison")
            mode = "sdr"
        else:
            gainmap, metadata = gainmap_result
            if verbose:
                print(f"  Gain map: min={metadata['gain_map_min']:.3f}, max={metadata['gain_map_max']:.3f}")

            # Reconstruct full HDR
            script_hdr = reconstruct_hdr(script_sdr, gainmap, metadata)
            # Apple: convert PQ to linear (normalized to SDR white = 1.0)
            apple_linear = pq_to_linear(apple_img, normalize_to_sdr=True)

            script_stats = analyze_image("Script HDR (reconstructed from gain map)", script_hdr, verbose)
            apple_stats = analyze_image("Apple HDR (LINEAR from PQ)", apple_linear, verbose)

    if mode == "sdr":
        # SDR comparison: linearize both, clip Apple to SDR range for fair comparison
        script_linear = srgb_to_linear(script_sdr)
        apple_linear = pq_to_linear(apple_img, normalize_to_sdr=True)
        # Clip Apple's HDR to SDR range (1.0 = SDR white) for fair comparison
        apple_sdr = np.clip(apple_linear, 0, 1.0)

        script_stats = analyze_image("Script SDR (LINEAR)", script_linear, verbose)
        apple_stats = analyze_image("Apple SDR (LINEAR, clipped to 1.0)", apple_sdr, verbose)

    elif mode == "raw":
        # Raw encoded values (meaningless but kept for reference)
        script_stats = analyze_image("Script JXL (sRGB encoded)", script_sdr, verbose)
        apple_stats = analyze_image("Apple PNG (PQ encoded)", apple_img, verbose)

    # Compute comparison metrics
    ratio = script_stats['lum_mean'] / apple_stats['lum_mean'] if apple_stats['lum_mean'] > 0 else float('inf')
    median_ratio = script_stats['lum_median'] / apple_stats['lum_median'] if apple_stats['lum_median'] > 0 else float('inf')

    comparison = {
        "name": name,
        "script_lum_mean": script_stats['lum_mean'],
        "apple_lum_mean": apple_stats['lum_mean'],
        "ratio": ratio,
        "script_lum_median": script_stats['lum_median'],
        "apple_lum_median": apple_stats['lum_median'],
        "median_ratio": median_ratio,
    }

    if verbose:
        print("\n" + "-" * 40)
        print("KEY DIFFERENCES:")
        print("-" * 40)
        print(f"  Script mean lum: {script_stats['lum_mean']:.6f}")
        print(f"  Apple mean lum:  {apple_stats['lum_mean']:.6f}")
        print(f"  Ratio (Script/Apple): {ratio:.3f}x")
        print(f"  Script 50th percentile: {script_stats['lum_median']:.6f}")
        print(f"  Apple 50th percentile:  {apple_stats['lum_median']:.6f}")

    return comparison


def run_full_comparison(
    script_dir: Path = Path("/var/home/admin/Pictures/script-output"),
    apple_dir: Path = Path("/var/home/admin/Pictures/apple-output"),
    samples: list[str] | None = None,
    verbose: bool = True,
    mode: str = "hdr"
) -> list[dict]:
    """Run comparison on all sample images.

    Args:
        script_dir: Directory containing script-generated JXL files
        apple_dir: Directory containing Apple reference PNG files
        samples: List of image names to compare (without extension)
        verbose: Print detailed statistics
        mode: Comparison mode - "hdr" (reconstruct from gain map), "sdr", or "raw"
    """
    print("\n" + "=" * 70)
    if mode == "hdr":
        print("HDR COMPARISON MODE (RECOMMENDED)")
        print("=" * 70)
        print("Reconstructing full HDR from SDR base + gain map:")
        print("  - Script: SDR base * gain_ratio = HDR")
        print("  - Apple:  Inverse PQ EOTF → linear HDR")
        print("This compares the ACTUAL HDR content!")
    elif mode == "sdr":
        print("SDR COMPARISON MODE")
        print("=" * 70)
        print("Comparing SDR base only (ignoring gain map):")
        print("  - Script: SDR base (linearized)")
        print("  - Apple:  HDR (linearized)")
        print("NOTE: This ignores script's HDR content in gain map!")
    else:
        print("WARNING: RAW ENCODED VALUE COMPARISON (not recommended)")
        print("=" * 70)
        print("Comparing raw sRGB vs PQ encoded values - ratios are MEANINGLESS!")

    if samples is None:
        # Find all JXL files in script output
        samples = sorted([f.stem for f in script_dir.glob("*.jxl")])

    results = []

    for sample in samples:
        if verbose:
            print("\n" + "=" * 70)
            print(f"Comparing {sample}")
            print("=" * 70)

        jxl_path = script_dir / f"{sample}.jxl"
        png_path = apple_dir / f"{sample}.png"

        result = compare_images(jxl_path, png_path, sample, verbose, mode)
        if result:
            results.append(result)

    return results


def print_summary_table(results: list[dict]) -> None:
    """Print a summary table of all comparisons."""
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Image':<12} {'Script Lum':<12} {'Apple Lum':<12} {'Ratio':<10} {'Status'}")
    print("-" * 70)

    for r in results:
        ratio = r['ratio']
        if 0.9 <= ratio <= 1.1:
            status = "OK"
        elif ratio > 1.1:
            status = f"TOO BRIGHT (+{(ratio-1)*100:.0f}%)"
        else:
            status = f"TOO DARK ({(ratio-1)*100:.0f}%)"

        print(f"{r['name']:<12} {r['script_lum_mean']:<12.4f} {r['apple_lum_mean']:<12.4f} {ratio:<10.3f} {status}")

    print("-" * 70)

    # Overall statistics
    ratios = [r['ratio'] for r in results]
    print(f"Mean ratio: {np.mean(ratios):.3f}x")
    print(f"Std dev: {np.std(ratios):.3f}")
    print(f"Range: [{min(ratios):.3f}x, {max(ratios):.3f}x]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare script JXL output with Apple PNG reference images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  compare_images.py                           # HDR comparison (reconstructs from gain map)
  compare_images.py --mode sdr                # SDR base only (ignores gain map)
  compare_images.py --mode raw                # Raw encoded values (meaningless)
  compare_images.py --script /path/to/jxl --apple /path/to/png

Comparison Modes:
  hdr (default): Reconstruct full HDR from SDR base + gain map, compare with Apple HDR
  sdr:           Compare SDR base only (linearized), ignoring HDR gain map
  raw:           Compare raw encoded pixels (sRGB vs PQ - meaningless!)

The script JXL stores HDR as: SDR base (sRGB) + gain map
Apple PNG stores HDR as: PQ-encoded pixels (full HDR in pixel values)
"""
    )
    parser.add_argument("--script", "-s", type=Path,
                       default=Path("/var/home/admin/Pictures/script-output"),
                       help="Directory containing script-generated JXL files")
    parser.add_argument("--apple", "-a", type=Path,
                       default=Path("/var/home/admin/Pictures/apple-output"),
                       help="Directory containing Apple reference PNG files")
    parser.add_argument("--mode", "-m", choices=["hdr", "sdr", "raw"], default="hdr",
                       help="Comparison mode: hdr (default), sdr, or raw")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Only show summary table")

    args = parser.parse_args()

    results = run_full_comparison(
        script_dir=args.script,
        apple_dir=args.apple,
        mode=args.mode,
        verbose=not args.quiet
    )
    if results:
        print_summary_table(results)
