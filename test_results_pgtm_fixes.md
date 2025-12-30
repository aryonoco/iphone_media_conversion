# Test Results: PGTM Bug Fixes

## Run: 2025-12-30 23:25 UTC

## Changes Made

1. **Weight-to-Index Mapping**: Changed from `weight * (N-1)` to `weight * N` in both `_apply_profile_gain_table_map()` and `_compute_pgtm_gains()`

2. **HDR Path Weight Order**: Changed to compute weight from raw pixels with exposure scaling, then apply exposure after PGTM (per DNG SDK order)

## Results: NO SIGNIFICANT CHANGE

### Summary Statistics

| Metric | Before Fixes | After Fixes | Change |
|--------|--------------|-------------|--------|
| Mean ratio | 1.165x | 1.164x | -0.001 |
| Std dev | 0.277 | 0.276 | -0.001 |
| Range | [0.775x, 1.656x] | [0.774x, 1.654x] | ~same |

### Per-Image Comparison

| Image | Before | After | Change |
|-------|--------|-------|--------|
| IMG_2208 | 0.934x | 0.934x | 0.000 |
| IMG_2211 | 1.044x | 1.043x | -0.001 |
| IMG_2212 | 1.043x | 1.042x | -0.001 |
| IMG_2213 | 1.292x | 1.292x | 0.000 |
| IMG_2219 | 1.499x | 1.499x | 0.000 |
| IMG_2220 | 1.077x | 1.076x | -0.001 |
| IMG_2221 | 0.775x | 0.774x | -0.001 |
| IMG_2222 | 1.656x | 1.654x | -0.002 |

## Analysis: Why No Change?

### 1. Weight-to-Index Change is Minimal
The difference between `weight * 255` and `weight * 256` is very small:
- At weight=0.5: index shifts from 127.5 to 128
- With smooth interpolation, this barely affects the output

### 2. HDR Path Changes Don't Affect SDR Comparison
The HDR path changes affect gain map computation, but we're comparing the SDR base image. The SDR path already had correct exposure scaling in `_apply_profile_gain_table_map()`.

### 3. Fundamental Comparison Problem
**The comparison is comparing different signal encodings:**

- Apple PNG: **PQ (SMPTE ST 2084)** transfer function
- Script JXL: **sRGB** transfer function

When decoded to raw pixels:
- PQ encodes luminance with a perceptual curve designed for 0-10,000 nits
- sRGB encodes luminance with gamma ~2.2 designed for 0-80 nits

A direct pixel comparison between PQ and sRGB values is not meaningful. The same scene luminance produces very different pixel values in each encoding.

## Conclusion

The PGTM fixes are technically correct per DNG SDK, but they don't solve the visual difference problem because:

1. The bug impact was smaller than expected (smooth interpolation masks the difference)
2. The real issue may be the fundamental architecture difference:
   - Apple outputs true HDR (PQ encoding, full dynamic range in pixels)
   - Script outputs SDR + gain map (sRGB base, HDR reconstructed via gain map)

## Next Steps to Consider

1. **Better comparison method**: Convert both to linear before comparing, or use a perceptual metric (SSIM, dE2000)

2. **Visual inspection**: Actually view the images on an HDR display to see if they look similar

3. **Check if gain map works**: The JXL has a gain map - does it reconstruct HDR correctly when viewed in an HDR-aware viewer?

4. **Alternative architecture**: Consider outputting PQ directly instead of sRGB + gain map to match Apple's approach
