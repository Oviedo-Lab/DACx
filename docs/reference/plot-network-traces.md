# Plot spike traces for network from SGT simulation

This function plots spike traces for a network object from a Spatial
Growth-Transform (SGT) simulation.

## Usage

``` r
plot.network.traces(network, return_plot)
```

## Arguments

- network:

  Network object with SGT simulation traces to plot.

- return_plot:

  Logical indicating whether to return the ggplot object (TRUE) or print
  it (FALSE) (default: FALSE).

## Value

A ggplot object showing spike traces for all neurons in the network over
time.
