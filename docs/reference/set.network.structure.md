# Set network structure

This function sets the structure of a network object, defining its
layers, columns, neuron types, and local connectivity parameters. It
also generates local nodes based on the specified structure.

## Usage

``` r
set.network.structure(
  network,
  neuron_types = c("principal"),
  hemisphere_names = NULL,
  subcortical_layer_names = NULL,
  layer_names = c("layer"),
  n_hemispheres = 1,
  n_subcortical_layers = 0,
  n_layers = 1,
  n_columns = 1,
  n_patches = 1,
  layer_height = 180,
  column_diameter = 120,
  segment_length = 20,
  hem_separation_factor = 40,
  sub_separation_factor = 20,
  layer_separation_factor = 2.5,
  column_separation_factor = 2.5,
  patch_separation_factor = 2.5,
  local_synaptic_conductance = 0.1,
  subcortical_local_conductance = 0.1,
  synaptic_neighborhood = 10,
  neurons_per_node = 10
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

- hemisphere_names:

  Character vector of length n_hemispheres giving names for the
  hemispheres (default: auto-generated as "left" or c("left","right")).

- subcortical_layer_names:

  Character vector of length n_subcortical_layers giving names for the
  subcortical layers (default: auto-generated as "subL1", "subL2", ...).
  Must be distinct from all cortical layer names.

- layer_names:

  Character vector giving names of cortical layers in the network,
  ordered deepest to most superficial, e.g. c("L6", "L5", "L4", "L3",
  "L2", "L1").

- n_hemispheres:

  Integer giving number of hemispheres; must be 1 or 2 (default: 1).

- n_subcortical_layers:

  Integer giving number of subcortical layers (e.g., thalamic relay
  nuclei); can be 0 (default: 0).

- n_layers:

  Integer giving number of cortical layers in the network.

- n_columns:

  Integer giving number of columns in the network.

- n_patches:

  Integer giving the number of "patches" (n_layers x n_columns sheets)
  in the network.

- layer_height:

  Numeric giving height of each layer (default value is 180.0, which
  assumes an implicit unit of micron).

- column_diameter:

  Numeric giving diameter of each column (default value is 120.0, which
  assumes an implicit unit of micron).

- segment_length:

  Numeric giving expected length of each segment in the axonal and
  dendritic processes of each neuron (default value is 20.0, which
  assumes an implicit unit of micron).

- hem_separation_factor:

  Numeric giving distance between hemispheres as a fraction of column
  diameter (default: 5.0).

- sub_separation_factor:

  Numeric giving distance from the cortical sheet to the first
  subcortical layer as a fraction of layer height (default: 5.0).

- layer_separation_factor:

  Numeric giving mean distance between layers as a fraction of layer
  height (default: 2.5).

- column_separation_factor:

  Numeric giving mean distance between columns as a fraction of column
  diameter (default: 2.5).

- patch_separation_factor:

  Numeric giving mean distance between network patches as a fraction of
  column diameter (default: 2.5).

- local_synaptic_conductance:

  List (one entry per cortical layer) of matrices giving synaptic
  conductance in nS for local connections by cell-type; each matrix must
  have dimensions matching length of neuron_types (rows and columns).

- subcortical_local_conductance:

  List (one entry per subcortical layer) of matrices giving synaptic
  conductance for local connections in each subcortical layer. Required
  if n_subcortical_layers \> 0; can be a single matrix or scalar
  (broadcast to all layers).

- synaptic_neighborhood:

  Numeric giving the radius (in microns) within which an axon node will
  trigger a synapse when near a dendrite node (default: 10.0).

- neurons_per_node:

  Matrix giving mean number of neurons of each type per node in each
  layer, with cortical layers first and then subcortical layers;
  dimensions must match n_layers + n_subcortical_layers (rows) and
  length of neuron_types (columns), or 2 \* (n_layers +
  n_subcortical_layers) if specifying different cell type counts for a
  second hemisphere. If there are two hemispheres but only n_layers +
  n_subcortical_layers rows, then the counts are reused for the second
  hemisphere.

## Value

The updated network object with the specified structure and local nodes
generated.
