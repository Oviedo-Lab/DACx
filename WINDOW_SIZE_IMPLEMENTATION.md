# Tutorial Implementation: Window Size Analysis for Phase Planes

## Summary of Changes

Updated `vignettes/tutorial_SGT.Rmd` to implement recommendations about temporal window sizes in phase plane analysis.

## What Changed

### 1. **Made window size configurable** (line 126)
```r
window_ms <- 100         # temporal window for computing firing rates (ms)
```
Users can now easily adjust this parameter to explore different temporal resolutions.

### 2. **Added explanatory comments** (lines 121-125)
```r
# ── window size for phase plane ────────────────────────────────────────
# NOTE: The window size determines the temporal resolution of the phase plane.
# - Larger windows (500ms): shows steady-state behavior, averaged dynamics
# - Smaller windows (100ms): captures transient dynamics and oscillatory cycles
# For networks with oscillations, use smaller windows to see limit cycle structure.
```

### 3. **Updated phase plane title** (line 187)
Now displays the window size being used:
```r
title = sprintf("SGT phase plane (%dms windows)", window_ms)
```

### 4. **Added new section: "Window size and phase plane resolution"** (lines 192-271)
This new section:
- Explains why window size matters
- Shows that smaller windows (100ms) reveal transient dynamics
- Shows that larger windows (500ms) smooth to steady-state
- Includes side-by-side comparison plots
- Provides guidance on when to use each window size

### 5. **Added comparison code block** (lines 198-262)
New `compare_window_sizes` chunk that:
- Recomputes the phase plane with 500ms windows
- Plots both resolutions side-by-side
- Visually demonstrates the smoothing effect of larger windows

## Key Changes to Parameter Defaults

| Parameter | Old Value | New Value | Reason |
|-----------|-----------|-----------|--------|
| `half_ms` | 500 ms | 100 ms (adjustable) | Finer temporal resolution |
| `full_ms` | 1000 ms | 200 ms (2× window) | Maintains half/full ratio |
| `window_ms` | N/A (hardcoded) | 100 ms (configurable) | New parameter for easy adjustment |

## Rationale

**Problem identified:** The original 500ms windows averaged away oscillatory structure in networks with ~5-10 Hz oscillations. A 500ms window would contain 2.5-5 complete oscillation cycles, resulting in heavily smoothed firing rates that don't reveal the true phase-space dynamics.

**Solution:** 
- Default to 100ms windows (1-2 cycles for 5-10 Hz oscillations)
- Provide side-by-side comparison with 500ms windows
- Make window size easily adjustable
- Explain the trade-offs clearly

## Expected Outcomes When Running Updated Tutorial

1. **Phase plane shows more structure** with 100ms windows
2. **Side-by-side comparison reveals** how larger windows smooth transients
3. **Users understand the parameter choice** and can adjust for their own networks
4. **Better diagnostic tool** for identifying oscillatory vs. steady-state behavior

## Guidance for Users

The tutorial now includes clear guidance:

- **For oscillatory networks (5-10 Hz):** Use 50-100ms windows
- **For steady-state analysis:** Use 500ms+ windows
- **Default behavior:** Uses 100ms windows as a balance for typical cortical networks

Users can now easily modify `window_ms` at the top of the code block to adjust temporal resolution based on their specific network's expected dynamics.

## References

- Oscillation detection via phase plane analysis requires temporal resolution matching system timescale
- Nyquist-Shannon sampling principle: need at least 2 samples per cycle
- For 10 Hz oscillations: minimum window ~50ms for adequate sampling
