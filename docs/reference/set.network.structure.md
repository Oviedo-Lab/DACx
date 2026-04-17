# Set network structure

This function sets the structure of a network object, defining its
layers, columns, neuron types, and local connectivity parameters. It
also generates local nodes based on the specified structure.

## Usage

``` r
set.network.structure(
  network,
  neuron_types = c("principal"),
  layer_names = c("layer"),
  n_layers = 1,
  n_columns = 1,
  patch_depth = 1,
  layer_height = 180,
  column_diameter = 120,
  segment_length = 20,
  layer_separation_factor = 2.5,
  column_separation_factor = 2.5,
  patch_separation_factor = 2.5,
  neurons_per_node = 10,
  local_synaptic_conductance = 1e-10,
  synaptic_neighborhood = 10
)
```

## Arguments

- network:

  Network object to configure.

- neuron_types:

  Character vector giving types of neurons in the network. Known types
  can be accessed using
  [`print.known.celltypes()`](https://Oviedo-Lab.org/DACx/reference/print-known-celltypes.md).
  Default is "principal", which will assign the most common neuron type
  for each layer, as defined in
  [`principal.neurons()`](https://Oviedo-Lab.org/DACx/reference/principal.neurons.md).

- layer_names:

  Character vector giving names of layers in the network, ordered
  deepest to most superficial, e.g. c("L6", "L5", "L4", "L3", "L2",
  "L1").

- n_layers:

  Integer giving number of layers in the network.

- n_columns:

  Integer giving number of columns in the network.

- patch_depth:

  Integer giving the number of "patches" (n_layers x n_columns sheets)
  in the network.

- layer_height:

  Numeric giving height of each layer (in units specified at network
  creation, default unit is microns, default value is 180.0).

- column_diameter:

  Numeric giving diameter of each column (in units specified at network
  creation, default unit is microns, default value is 120.0).

- segment_length:

  Numeric giving expected length of each segment in the axonal and
  dendritic processes of each neuron (in units specified at network
  creation, default unit is microns, default value is 20.0).

- layer_separation_factor:

  Numeric giving mean distance between layers as a fraction of layer
  height (default: 2.5).

- column_separation_factor:

  Numeric giving mean distance between columns as a fraction of column
  diameter (default: 2.5).

- patch_separation_factor:

  Numeric giving mean distance between network patches as a fraction of
  column diameter (default: 2.5).

- neurons_per_node:

  Matrix giving number of neurons of each type per node in each layer;
  dimensions must match n_layers (rows) and length of neuron_types
  (columns).

- local_synaptic_conductance:

  List (one entry per layer) of matrices giving synaptic conductance in
  millisiemens for local connections by cell-type; each matrix must have
  dimensions matching length of neuron_types (rows and columns).

- synaptic_neighborhood:

  Numeric giving the radius (in microns) within which an axon node will
  trigger a synapse when near a dendrite node (default: 10.0).

- neuron_type_valences:

  Numeric vector giving valences of each neuron type, e.g. c(1, -1) for
  excitatory and inhibitory neurons.

- neuron_type_temporal_modulation:

  Numeric matrix giving temporal modulation time components (for
  modulation time in the unit_time of the network) for each neuron type:
  bias, step size, and count cutoff (rows as neuron types, columns as
  components). Will example a single value or a vector of length three.

## Value

The updated network object with the specified structure and local nodes
generated.
