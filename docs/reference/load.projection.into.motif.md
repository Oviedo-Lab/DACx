# Load projection into motif

This function loads a projection schema into a motif object. Projections
define internode connectivity within a network built using the motif.

## Usage

``` r
load.projection.into.motif(
  motif,
  presynaptic_layer,
  postsynaptic_layer,
  pre_neuron_fraction = 0.5,
  presynaptic_type = "principal",
  postsynaptic_type = "principal",
  max_col_shift_up = 0,
  max_col_shift_down = 0,
  max_pch_shift_up = 0,
  max_pch_shift_down = 0,
  hem_shift = 0L,
  via_apical = FALSE
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

- pre_neuron_fraction:

  Numeric between 0 and 1 giving the fraction of eligible presynaptic
  neurons that send axons in this projection (default: 0.5). This
  controls projection sparsity; conductance values are automatically
  looked up from neuron type properties.

- presynaptic_type:

  Character string giving type of presynaptic neuron (default:
  "principal").

- postsynaptic_type:

  Character string giving type of postsynaptic neuron (default:
  "principal").

- max_col_shift_up:

  Maximum number of columns upwards (increasing columnar indexes) that
  the projection can reach (default: 0, should be positive integer).

- max_col_shift_down:

  Maximum number of columns downwards (decreasing columnar indexes) that
  the projection can reach (default: 0, should be positive integer).

- max_pch_shift_up:

  Same as `max_col_shift_up`, but for secondary columnar axis "patch".

- max_pch_shift_down:

  Same as `max_col_shift_down`, but for secondary columnar axis "patch".

- hem_shift:

  Hemisphere shift for the projection: 0 = same hemisphere (default), 1
  = contralateral hemisphere. Ignored when the network has only one
  hemisphere.

## Value

The updated motif object with the new projection loaded.
