# Plot spike traces for network from SGT simulation

This function plots spike traces for a network object from a Spatial
Growth-Transform (SGT) simulation.

## Usage

``` r
plot.network.traces(
  network, 
  return_plot = FALSE, 
  I_stim      = NULL, 
  window_size = 0.01, 
  plot_rates  = TRUE
)
```

## Arguments

- network:

  Network object with SGT simulation traces to plot.

- return_plot:

  Logical indicating whether to return the ggplot object (TRUE) or print
  it (FALSE) (default: FALSE).

- I_stim:

  Matrix of stimulus currents, with rows representing neurons and
  columns representing sample times. Presumably the one used to generate
  the traces. Options. If provided, will be added to the bottom of the
  plot.

- window_size:

  Proportion of time steps to use as a moving window for computing spike
  rate (default: 0.01).

- plot_rates:

  Boolean specifying whether to include a plot of estimated mean spike
  rate for each cell type above the type's membrane potential plot.

## Value

A ggplot object showing spike traces for all neurons in the network over
time.
