# Baseline Comparison: Script vs Apple Photos

## Run: 2025-12-30 22:45 UTC (Baseline - Before Changes)

## Summary Statistics

- Mean ratio: 1.160x
- Std dev: 0.276
- Range: [0.775x, 1.654x]

## Per-Image Results

| Image | BaselineExposure (EV) | Script Lum Mean | Apple Lum Mean | Ratio | Status |
|-------|----------------------|-----------------|----------------|-------|--------|
| IMG_2208 | +3.65 | 0.2296 | 0.2455 | 0.935x | OK |
| IMG_2211 | -0.50 | 0.3446 | 0.3302 | 1.044x | OK |
| IMG_2212 | +0.94 | 0.3209 | 0.3089 | 1.039x | OK |
| IMG_2213 | +4.88 | 0.4869 | 0.3821 | 1.274x | TOO BRIGHT (+27%) |
| IMG_2219 | +3.38 | 0.5689 | 0.3794 | 1.499x | TOO BRIGHT (+50%) |
| IMG_2220 | +1.47 | 0.3740 | 0.3527 | 1.060x | OK |
| IMG_2221 | +0.37 | 0.2461 | 0.3178 | 0.775x | TOO DARK (-23%) |
| IMG_2222 | +1.42 | 0.5059 | 0.3058 | 1.654x | TOO BRIGHT (+65%) |

## Observations

1. **5 of 8 images are within acceptable range** (0.9-1.1x)
2. **3 images are too bright**: IMG_2213, IMG_2219, IMG_2222
3. **1 image is too dark**: IMG_2221
4. The correlation with BaselineExposure is not straightforward:
   - IMG_2208 (high exposure +3.65) is slightly dark (0.935x)
   - IMG_2222 (moderate exposure +1.42) is very bright (1.654x)
5. There appears to be scene-dependent variation beyond just exposure compensation

## Test Configuration

- Script output: `/var/home/admin/Pictures/script-output/`
- Apple reference: `/var/home/admin/Pictures/apple-output/`
- Input DNGs: `/var/home/admin/Pictures/iphone-orig/`
- Test script: `compare_images.py`
