# Fourier Power Spectrum of Population Spike Rate

Recomputes the population-average spike rate trace for each cell type
(using the same moving-window method as `plot.network.traces`) and plots
the one-sided FFT power spectrum (power vs. frequency in Hz) for each
type.

## Usage

``` r
plot.network.spikerate.spectrum(
  network,
  return_plot = FALSE,
  window_size = 0.01,
  detrend     = TRUE,
  max_freq    = NULL
)
```

## Arguments

- network:

  Network object with BGT simulation traces.

- return_plot:

  Logical; if `TRUE` return the ggplot/patchwork object instead of
  printing it (default: `FALSE`).

- window_size:

  Proportion of total simulation duration to use as the moving-window
  width when computing spike rate (default: 0.01, matching
  `plot.network.traces`).

- detrend:

  Logical; if `TRUE` (default) the mean is subtracted from each
  spike-rate trace before the FFT, removing the DC component so that the
  spectrum focuses on oscillatory content.

- max_freq:

  Optional upper frequency limit (Hz) for the x-axis. If `NULL`
  (default) the full range up to the Nyquist frequency is shown.

## Value

A patchwork/ggplot object (one panel per cell type), or `NULL` invisibly
when `return_plot = FALSE`.
