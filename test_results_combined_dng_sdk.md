# Test Results: Combined DNG SDK Changes

## Run: 2025-12-30 22:59 UTC

## Changes Made (All Together)

1. **Pipeline Order**: Changed from `Exposure → PGTM → ToneCurve` to `PGTM → Exposure → ToneCurve`
2. **PGTM Weight Scaling**: Changed from `1.0` to `exposure_mult` (= 2^BaselineExposure)
3. **PGTM_STRENGTH**: Changed from `0.5` to `1.0`

## Results: NEARLY IDENTICAL TO BASELINE

### Summary Statistics

| Metric | Baseline | Combined DNG SDK | Change |
|--------|----------|------------------|--------|
| Mean ratio | 1.160x | 1.165x | +0.005 |
| Std dev | 0.276 | 0.277 | +0.001 |
| Range | [0.775x, 1.654x] | [0.775x, 1.656x] | ~same |

### Per-Image Comparison

| Image | BaselineExp | Baseline Ratio | Combined DNG SDK | Change |
|-------|-------------|----------------|------------------|--------|
| IMG_2208 | +3.65 EV | 0.935x | 0.934x | -0.001 |
| IMG_2211 | -0.50 EV | 1.044x | 1.044x | 0.000 |
| IMG_2212 | +0.94 EV | 1.039x | 1.043x | +0.004 |
| IMG_2213 | +4.88 EV | 1.274x | 1.292x | +0.018 |
| IMG_2219 | +3.38 EV | 1.499x | 1.499x | 0.000 |
| IMG_2220 | +1.47 EV | 1.060x | 1.077x | +0.017 |
| IMG_2221 | +0.37 EV | 0.775x | 0.775x | 0.000 |
| IMG_2222 | +1.42 EV | 1.654x | 1.656x | +0.002 |

### Analysis

The combined DNG SDK changes result in nearly identical output to baseline. This suggests:

1. The three changes (order, weight scaling, PGTM_STRENGTH) interact in complex ways
2. The effects partially cancel each other out
3. Simply "matching DNG SDK" doesn't automatically produce better results

### Key Observations

The problematic images remain the same:
- **Too Bright**: IMG_2213 (+29%), IMG_2219 (+50%), IMG_2222 (+66%)
- **Too Dark**: IMG_2221 (-23%)

There's no clear correlation with BaselineExposure:
- IMG_2208 has high exposure (+3.65 EV) but is correctly exposed (0.934x)
- IMG_2221 has low exposure (+0.37 EV) but is too dark (0.775x)
- IMG_2222 has medium exposure (+1.42 EV) but is way too bright (1.656x)

### Hypothesis

The variance may be due to:
1. Per-image PGTM content differences
2. Per-image tone curve differences
3. Transfer function mismatch (sRGB vs PQ)
4. Something else entirely (color matrix, white balance, etc.)

## Files Modified

- Line 108: `PGTM_STRENGTH: float = 1.0`
- Lines 1366-1387: Swapped order of PGTM and Exposure application
- Line 1372: Using `exposure_mult` as PGTM weight
