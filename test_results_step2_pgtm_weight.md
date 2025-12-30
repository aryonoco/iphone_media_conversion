# Test Results: Step 2 - PGTM Weight Scaling

## Run: 2025-12-30 22:53 UTC

## Hypothesis

Line 1378 passes `1.0` to `_apply_profile_gain_table_map()` but DNG SDK `dng_render.cpp:1887` uses `pow(2.0, fBaselineExposure)`.

## Change Made

```python
# Before:
_apply_profile_gain_table_map(temp_ppm, pgtm, 1.0)

# After:
_apply_profile_gain_table_map(temp_ppm, pgtm, exposure_mult)  # where exposure_mult = 2^baseline_exposure_ev
```

## Results: WORSE - Reverted

### Summary Statistics

| Metric | Baseline | After Fix | Verdict |
|--------|----------|-----------|---------|
| Mean ratio | 1.160x | 0.840x | Worse (too dark now) |
| Std dev | 0.276 | 0.225 | Better consistency but wrong direction |
| Range | [0.775x, 1.654x] | [0.345x, 1.090x] | Shifted dark |

### Per-Image Comparison

| Image | BaselineExp | Baseline Ratio | After Fix | Change |
|-------|-------------|----------------|-----------|--------|
| IMG_2208 | +3.65 EV | 0.935x (OK) | 0.345x | **MUCH WORSE** (-63%) |
| IMG_2211 | -0.50 EV | 1.044x (OK) | 1.090x | Slightly worse |
| IMG_2212 | +0.94 EV | 1.039x (OK) | 0.933x | Slightly worse |
| IMG_2213 | +4.88 EV | 1.274x (Bright) | 0.849x | Better (was too bright) |
| IMG_2219 | +3.38 EV | 1.499x (Bright) | 1.017x | **IMPROVED** |
| IMG_2220 | +1.47 EV | 1.060x (OK) | 0.717x | Worse |
| IMG_2221 | +0.37 EV | 0.775x (Dark) | 0.747x | Slightly worse |
| IMG_2222 | +1.42 EV | 1.654x (Bright) | 1.022x | **IMPROVED** |

### Analysis

- The fix helped images that were too bright (IMG_2213, IMG_2219, IMG_2222)
- But it made images that were OK or dark much worse
- IMG_2208 went from 0.935x to 0.345x (extremely dark) despite having high BaselineExposure
- The interaction between PGTM weight scaling and the rest of the pipeline is complex

### Conclusion

**Reverted change.** The DNG SDK reference may apply weight scaling differently in context (e.g., different pipeline order, different exposure handling). The previous value of `1.0` works better empirically for the current pipeline.

## Files Modified

- `/var/home/admin/projects/iphone_media_conversion/export_photos.py` line 1378-1380 (reverted with comment)
