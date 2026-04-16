# Load projection into motif

This function loads a projection schema into a motif object. Projections
define internode connectivity within a network built using the motif.

## Usage

``` r
load.projection.into.motif(
  motif,
  presynaptic_layer,
  postsynaptic_layer,
  connection_strength = 0.5,
  presynaptic_type = "principal",
  postsynaptic_type = "principal",
  max_col_shift_up = 0,
  max_col_shift_down = 0
)
```

## Arguments

- motif:

  Motif object into which to load the projection.

- presynaptic_layer:

  Character string giving layer of presynaptic neuron, e.g. "L1", "L2",
  "L3", "L4", etc.

- postsynaptic_layer:

  Character string, or vector of character strings, giving layer of
  postsynaptic neuron.

- connection_strength:

  Numeric giving overall strength of the projection (default: 0.5).

- presynaptic_type:

  Character string giving type of presynaptic neuron, e.g. "excitatory",
  "inhibitory", etc. (default: "principal").

- postsynaptic_type:

  Character string giving type of postsynaptic neuron, e.g.
  "excitatory", "inhibitory", etc. (default: "principal").

- max_col_shift_up:

  Maximum number of columns upwards (increasing columnar indexes) that
  the projection can reach (default: 0, should be positive integer).

- max_col_shift_down:

  Maximum number of columns downwards (decreasing columnar indexes) that
  the projection can reach (default: 0, should be positive integer).

## Value

The updated motif object with the new projection loaded.
