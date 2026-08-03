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

- input_matrix:

  Matrix of stimulus currents, with rows representing neurons and
  columns representing sample times. Presumably the one used to generate
  the traces. Options. If provided, will be added to the bottom of the
  plot.

## Value

A ggplot object showing spike traces for all neurons in the network over
time.
