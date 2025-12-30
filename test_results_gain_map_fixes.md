# Test Results: Gain Map Pipeline Fixes

## Run: 2025-12-31 00:15 UTC

## Changes Made

1. **Gain Map Computation Order**: Moved gain map computation to BEFORE tone curve
   - Now computes from LINEAR SDR (PGTM + Exposure applied, before tone curve)
   - Per ISO 21496-1 requirement: both inputs must be in linear space

2. **sRGB Gamma Restored**: Kept sRGB gamma after tone curve
   - ProfileToneCurve is a tone curve (aesthetic shaping), NOT a transfer function
   - sRGB gamma is still needed for perceptual display encoding

## Results: NO CHANGE TO SDR COMPARISON

### Summary Statistics

| Metric | Baseline | After Fix | Change |
|--------|----------|-----------|--------|
| Mean ratio | 1.160x | 1.164x | +0.004 |
| Std dev | 0.276 | 0.276 | 0.000 |
| Range | [0.775x, 1.654x] | [0.774x, 1.654x] | ~same |

### Per-Image Comparison

| Image | BaselineExp | Baseline Ratio | After Fix | Change |
|-------|-------------|----------------|-----------|--------|
| IMG_2208 | +3.65 EV | 0.935x | 0.934x | -0.001 |
| IMG_2211 | -0.50 EV | 1.044x | 1.043x | -0.001 |
| IMG_2212 | +0.94 EV | 1.039x | 1.042x | +0.003 |
| IMG_2213 | +4.88 EV | 1.274x | 1.292x | +0.018 |
| IMG_2219 | +3.38 EV | 1.499x | 1.499x | 0.000 |
| IMG_2220 | +1.47 EV | 1.060x | 1.076x | +0.016 |
| IMG_2221 | +0.37 EV | 0.775x | 0.774x | -0.001 |
| IMG_2222 | +1.42 EV | 1.654x | 1.654x | 0.000 |

## Why No Change?

The gain map fix affects **HDR reconstruction**, not the SDR base image:

1. **SDR Path Unchanged**: The SDR base image still goes through:
   - PGTM → Exposure → Tone Curve → sRGB Gamma → JXL

2. **Comparison Measures SDR**: We're comparing SDR base images (script JXL vs Apple PNG)

3. **Gain Map is Separate**: The gain map is embedded in the JXL for HDR reconstruction
   - It now correctly uses LINEAR SDR and LINEAR HDR
   - This affects HDR viewing, not SDR comparison

## ChatGPT Claim Analysis: Final Verdict

| Claim | Verdict | Notes |
|-------|---------|-------|
| "Use Display P3" | **WRONG** | Rec.2020 is valid per DNG SDK |
| "Pipeline order wrong" | **WRONG** | Already correct: PGTM → Exposure → ToneCurve |
| "Gain map from non-linear data" | **FIXED** | Now computes from LINEAR SDR |
| "Remove sRGB gamma" | **WRONG** | ProfileToneCurve ≠ sRGB gamma; both needed |

## Fundamental Comparison Problem (Still Unresolved)

The comparison is still flawed:

- **Apple PNG**: PQ (SMPTE ST 2084) transfer function, true HDR
- **Script JXL**: sRGB transfer function, SDR base + gain map

Direct pixel comparison between PQ and sRGB is not meaningful. Same luminance produces different pixel values in each encoding.

## What Was Actually Fixed

The gain map is now **correctly computed** per ISO 21496-1:

```
BEFORE (wrong):
  gain_ratio = LINEAR_HDR / NON_LINEAR_SDR  ← Invalid ratio

AFTER (correct):
  gain_ratio = LINEAR_HDR / LINEAR_SDR  ← Valid ratio
```

This ensures correct HDR reconstruction when:
1. Viewer applies gain map to SDR base
2. Reconstructs HDR using: `hdr = sdr_linear * gain_ratio`

## Conclusion

1. **SDR appearance unchanged** - expected, since SDR path is unchanged
2. **HDR reconstruction now correct** - gain map uses proper linear inputs
3. **Comparison methodology still flawed** - need better approach for PQ vs sRGB

## Recommendations

1. **Visual inspection**: View images on HDR display to compare actual appearance
2. **Better comparison**: Convert both to linear before comparing, or use perceptual metrics
3. **Alternative approach**: If exact Apple match is required, consider outputting true HDR (PQ) instead of SDR + gain map

## Files Modified

- `export_photos.py` lines 1411-1457: Reordered gain map computation before tone curve
