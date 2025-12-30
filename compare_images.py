#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=1.26"]
# ///

"""Compare pixel values between script JXL and Apple PNG outputs.

Test infrastructure for incremental improvements to export_photos.py.
Compares luminance statistics between script output and Apple reference.
"""

import subprocess
import numpy as np
import os
from pathlib import Path

# Set library path for libjxl
os.environ["LD_LIBRARY_PATH"] = "/var/home/admin/projects/iphone_media_conversion/build/deps/lib64:/home/linuxbrew/.linuxbrew/lib"

DJXL = "/var/home/admin/projects/iphone_media_conversion/build/deps/bin/djxl"


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


def compare_images(script_jxl: Path, apple_png: Path, name: str, verbose: bool = True) -> dict | None:
    """Compare a single image pair and return comparison metrics."""
    import tempfile

    if not script_jxl.exists():
        print(f"  Script JXL not found: {script_jxl}")
        return None

    if not apple_png.exists():
        print(f"  Apple PNG not found: {apple_png}")
        return None

    # Decode script JXL
    with tempfile.NamedTemporaryFile(suffix='.ppm', delete=False) as f:
        ppm_path = Path(f.name)

    if not decode_jxl_to_ppm(script_jxl, ppm_path):
        print(f"  Failed to decode {script_jxl}")
        return None

    script_img = read_ppm(ppm_path)
    os.unlink(ppm_path)

    # Read Apple PNG
    apple_img = read_png(apple_png)
    if apple_img is None:
        return None

    # Analyze both
    script_stats = analyze_image("Script JXL (decoded)", script_img, verbose)
    apple_stats = analyze_image("Apple PNG (HDR PQ)", apple_img, verbose)

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
    verbose: bool = True
) -> list[dict]:
    """Run comparison on all sample images."""

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

        result = compare_images(jxl_path, png_path, sample, verbose)
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
    results = run_full_comparison()
    if results:
        print_summary_table(results)
