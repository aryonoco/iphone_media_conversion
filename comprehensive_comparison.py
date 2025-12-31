#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=1.26"]
# ///
"""Comprehensive comparison of script JXL vs Apple PNG exports for all 8 images."""
import subprocess
import numpy as np
import os
import tempfile

os.environ["LD_LIBRARY_PATH"] = "/var/home/admin/projects/iphone_media_conversion/build/deps/lib64:/home/linuxbrew/.linuxbrew/lib"
DJXL = "/var/home/admin/projects/iphone_media_conversion/build/deps/bin/djxl"
SDR_WHITE = 203.0

def pq_to_linear(pq):
    """SMPTE ST 2084 EOTF (PQ to linear)."""
    m1, m2 = 0.1593017578125, 78.84375
    c1, c2, c3 = 0.8359375, 18.8515625, 18.6875
    pq = np.clip(pq.astype(np.float64), 0, 1)
    pq_m2 = np.power(pq + 1e-12, 1.0 / m2)
    num = np.maximum(pq_m2 - c1, 0.0)
    denom = c2 - c3 * pq_m2
    return np.power(num / (denom + 1e-12), 1.0 / m1) * 10000.0 / SDR_WHITE

def load_image(path, is_jxl=False):
    """Load image and return both raw PQ values and linear values."""
    with tempfile.NamedTemporaryFile(suffix='.ppm', delete=False) as f:
        ppm = f.name
    if is_jxl:
        subprocess.run([DJXL, path, ppm, '--bits_per_sample=16'], check=True, capture_output=True)
    else:
        subprocess.run(['convert', path, '-depth', '16', f'ppm:{ppm}'], check=True, capture_output=True)
    with open(ppm, 'rb') as f:
        f.readline(); dims = f.readline().decode(); f.readline()
        w, h = map(int, dims.strip().split())
        data = np.frombuffer(f.read(), dtype='>u2').reshape(h, w, 3)
    os.unlink(ppm)
    pq = data.astype(np.float32) / 65535.0
    linear = pq_to_linear(pq)
    return pq, linear

def rgb_to_hsv(rgb):
    """Convert RGB to HSV."""
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    s = np.where(maxc > 0, (maxc - minc) / (maxc + 1e-10), 0)

    delta = maxc - minc + 1e-10
    h = np.zeros_like(r)

    # Red is max
    mask = (maxc == r) & (delta > 1e-10)
    h[mask] = ((g[mask] - b[mask]) / delta[mask]) % 6

    # Green is max
    mask = (maxc == g) & (delta > 1e-10)
    h[mask] = (b[mask] - r[mask]) / delta[mask] + 2

    # Blue is max
    mask = (maxc == b) & (delta > 1e-10)
    h[mask] = (r[mask] - g[mask]) / delta[mask] + 4

    h = h / 6.0  # Normalize to 0-1
    return np.stack([h, s, v], axis=-1)

def analyze_image(pq, linear, name):
    """Compute various metrics for an image."""
    # PQ space metrics (perceptual)
    pq_means = [np.mean(pq[:,:,i]) for i in range(3)]
    pq_rb = pq_means[0] / pq_means[2] if pq_means[2] > 0 else 0
    pq_brightness = np.mean(pq)

    # Linear space metrics
    lin_means = [np.mean(linear[:,:,i]) for i in range(3)]
    lin_rb = lin_means[0] / lin_means[2] if lin_means[2] > 0 else 0
    lin_brightness = np.mean(linear)

    # Contrast (std dev in PQ space - perceptually relevant)
    pq_contrast = np.std(np.mean(pq, axis=2))

    # HSV analysis (in PQ space for perceptual relevance)
    hsv = rgb_to_hsv(pq)
    avg_hue = np.mean(hsv[:,:,0])
    avg_sat = np.mean(hsv[:,:,1])

    # Color temperature proxy (R-B difference)
    color_temp = pq_means[0] - pq_means[2]

    # Highlight and shadow analysis
    lum_pq = 0.2126 * pq[:,:,0] + 0.7152 * pq[:,:,1] + 0.0722 * pq[:,:,2]
    shadow_mask = lum_pq < 0.3
    highlight_mask = lum_pq > 0.7

    shadow_brightness = np.mean(pq[shadow_mask]) if np.any(shadow_mask) else 0
    highlight_brightness = np.mean(pq[highlight_mask]) if np.any(highlight_mask) else 0

    return {
        'pq_r': pq_means[0], 'pq_g': pq_means[1], 'pq_b': pq_means[2],
        'pq_rb': pq_rb, 'pq_brightness': pq_brightness, 'pq_contrast': pq_contrast,
        'lin_r': lin_means[0], 'lin_g': lin_means[1], 'lin_b': lin_means[2],
        'lin_rb': lin_rb, 'lin_brightness': lin_brightness,
        'avg_hue': avg_hue, 'avg_sat': avg_sat,
        'color_temp': color_temp,
        'shadow_brightness': shadow_brightness,
        'highlight_brightness': highlight_brightness,
    }

def compare_metrics(script_m, apple_m):
    """Compare two sets of metrics."""
    return {
        'pq_rb_diff': (script_m['pq_rb'] / apple_m['pq_rb'] - 1) * 100 if apple_m['pq_rb'] > 0 else 0,
        'pq_brightness_diff': (script_m['pq_brightness'] / apple_m['pq_brightness'] - 1) * 100,
        'pq_contrast_diff': (script_m['pq_contrast'] / apple_m['pq_contrast'] - 1) * 100,
        'lin_rb_diff': (script_m['lin_rb'] / apple_m['lin_rb'] - 1) * 100 if apple_m['lin_rb'] > 0 else 0,
        'lin_brightness_diff': (script_m['lin_brightness'] / apple_m['lin_brightness'] - 1) * 100,
        'hue_diff': (script_m['avg_hue'] - apple_m['avg_hue']) * 360,  # in degrees
        'sat_diff': (script_m['avg_sat'] / apple_m['avg_sat'] - 1) * 100 if apple_m['avg_sat'] > 0 else 0,
        'color_temp_diff': script_m['color_temp'] - apple_m['color_temp'],
        'shadow_diff': (script_m['shadow_brightness'] / apple_m['shadow_brightness'] - 1) * 100 if apple_m['shadow_brightness'] > 0 else 0,
        'highlight_diff': (script_m['highlight_brightness'] / apple_m['highlight_brightness'] - 1) * 100 if apple_m['highlight_brightness'] > 0 else 0,
    }

# Process all 8 images
images = ['IMG_2208', 'IMG_2211', 'IMG_2212', 'IMG_2213', 'IMG_2219', 'IMG_2220', 'IMG_2221', 'IMG_2222']
script_dir = '/var/home/admin/Pictures/script-output'
apple_dir = '/var/home/admin/Pictures/apple-output'

print("=" * 90)
print("COMPREHENSIVE COMPARISON: Script JXL vs Apple PNG")
print("=" * 90)

all_diffs = []

for img in images:
    script_path = f'{script_dir}/{img}.jxl'
    apple_path = f'{apple_dir}/{img}.png'

    if not os.path.exists(script_path):
        print(f"\n{img}: Script JXL not found, skipping")
        continue
    if not os.path.exists(apple_path):
        print(f"\n{img}: Apple PNG not found, skipping")
        continue

    print(f"\n{'='*40}")
    print(f"  {img}")
    print(f"{'='*40}")

    script_pq, script_lin = load_image(script_path, is_jxl=True)
    apple_pq, apple_lin = load_image(apple_path)

    script_m = analyze_image(script_pq, script_lin, f'{img}_script')
    apple_m = analyze_image(apple_pq, apple_lin, f'{img}_apple')
    diffs = compare_metrics(script_m, apple_m)
    all_diffs.append(diffs)

    print(f"\n  PQ Space (Perceptual):")
    print(f"    R/B Ratio:    Script={script_m['pq_rb']:.3f}  Apple={apple_m['pq_rb']:.3f}  Diff={diffs['pq_rb_diff']:+.1f}%")
    print(f"    Brightness:   Script={script_m['pq_brightness']:.4f}  Apple={apple_m['pq_brightness']:.4f}  Diff={diffs['pq_brightness_diff']:+.1f}%")
    print(f"    Contrast:     Script={script_m['pq_contrast']:.4f}  Apple={apple_m['pq_contrast']:.4f}  Diff={diffs['pq_contrast_diff']:+.1f}%")

    print(f"\n  Linear Space:")
    print(f"    R/B Ratio:    Script={script_m['lin_rb']:.3f}  Apple={apple_m['lin_rb']:.3f}  Diff={diffs['lin_rb_diff']:+.1f}%")
    print(f"    Brightness:   Script={script_m['lin_brightness']:.4f}  Apple={apple_m['lin_brightness']:.4f}  Diff={diffs['lin_brightness_diff']:+.1f}%")

    print(f"\n  Color Analysis:")
    print(f"    Avg Hue:      Script={script_m['avg_hue']*360:.1f}°  Apple={apple_m['avg_hue']*360:.1f}°  Diff={diffs['hue_diff']:+.1f}°")
    print(f"    Avg Sat:      Script={script_m['avg_sat']:.4f}  Apple={apple_m['avg_sat']:.4f}  Diff={diffs['sat_diff']:+.1f}%")
    print(f"    Color Temp:   Script={script_m['color_temp']:.4f}  Apple={apple_m['color_temp']:.4f}  Diff={diffs['color_temp_diff']:+.4f}")

    print(f"\n  Tonal Analysis:")
    print(f"    Shadows:      Script={script_m['shadow_brightness']:.4f}  Apple={apple_m['shadow_brightness']:.4f}  Diff={diffs['shadow_diff']:+.1f}%")
    print(f"    Highlights:   Script={script_m['highlight_brightness']:.4f}  Apple={apple_m['highlight_brightness']:.4f}  Diff={diffs['highlight_diff']:+.1f}%")

# Summary statistics
print("\n" + "=" * 90)
print("SUMMARY STATISTICS (Average across all images)")
print("=" * 90)

if all_diffs:
    avg_diffs = {}
    for key in all_diffs[0].keys():
        values = [d[key] for d in all_diffs]
        avg_diffs[key] = np.mean(values)
        std_diffs = np.std(values)
        print(f"  {key:25s}: Mean={avg_diffs[key]:+.2f}  Std={std_diffs:.2f}")

print("\n" + "=" * 90)
print("KEY OBSERVATIONS")
print("=" * 90)
print("""
Based on these measurements:
- pq_rb_diff > 0: Script is more RED/YELLOW than Apple
- pq_rb_diff < 0: Script is more BLUE than Apple
- pq_brightness_diff > 0: Script is BRIGHTER than Apple
- pq_contrast_diff < 0: Script has LESS CONTRAST than Apple
- sat_diff < 0: Script is LESS SATURATED than Apple
- shadow_diff > 0: Script shadows are BRIGHTER (lifted) than Apple
""")
