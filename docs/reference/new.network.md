# Initialize neuron network

This function initializes a new network object with specified
parameters. Networks are used to simulate two-dimensional cortical
patches (of layers and columns) using Growth Transform dynamical
systems.

## Usage

``` r
new.network(
  network_name = "not_provided",
  recording_name = "not_provided",
  type = "Growth_Transform",
  genotype = "WT",
  sex = "not_provided",
  hemi = "not_provided",
  region = "not_provided",
  age = "not_provided",
  unit_time = "ms",
  unit_sample_rate = "Hz",
  unit_potential = "mV",
  unit_current = "mA",
  unit_conductance = "mS",
  unit_distance = "micron",
  t_per_bin = 1,
  sample_rate = 10000
)
```

## Arguments

- network_name:

  Character string giving name of the network (default: "not_provided").

- recording_name:

  Character string giving name of the recording on which this network is
  based (default: "not_provided").

- type:

  Character string giving type of network; "Growth_Transform" is the
  only option available (default: "Growth_Transform").

- genotype:

  Character string giving genotype of the animal from which the modelled
  network comes, e.g. "WT", "KO", "MECP2", "transgenic", etc. (default:
  "not_provided").

- sex:

  Character string giving sex of the animal from which the modelled
  network comes (default: "not_provided").

- hemi:

  Character string giving hemisphere of the animal from which the
  modelled network comes, e.g. "left", "right" (default:
  "not_provided").

- region:

  Character string giving brain region of the animal from which the
  modelled network comes, e.g. "V1", "M1", "CA1", "PFC", etc. (default:
  "not_provided").

- age:

  Character string giving age of the animal from which the modelled
  network comes, e.g. "P0", "P7", "P14", "adult", etc. (default:
  "not_provided").

- unit_time:

  Character string giving unit of time for spike raster or other
  recording data on which the model is based or being compared, e.g.
  "ms", "s", etc. (default: "ms").

- unit_sample_rate:

  Character string giving unit of sample rate for recording data on
  which the model is based or being compared, e.g. "Hz", "kHz", etc.
  (default: "Hz").

- unit_potential:

  Character string giving unit of cell-membrane potential for recording
  data on which the model is based or being compared, e.g. "mV", "uV",
  etc. (default: "mV").

- unit_current:

  Character string giving unit of cell current for recording data on
  which the model is based or being compared, e.g. "mA", "uA", etc.
  (default: "mA").

- unit_conductance:

  Character string giving unit of axon and dendrite conductance for
  recording data on which the model is based or being compared, e.g.
  "mS", "uS", etc. (default: "mS").

- unit_distance:

  Character string giving unit of distance axon and dendrite
  measurements on which the model is based or being compared, e.g.
  "micron", "mm", etc. (default: "micron").

- t_per_bin:

  Time (in above units) per bin, e.g., 1 ms per bin (default: 10.0).

- sample_rate:

  Sample rate (in above units), e.g., 1e4 Hz (default: 1e4).

## Value

A new network object.

## Details

Mathematically, networks are points (representing neurons) connected by
directed edges. Within the growth-transform (GT) model framework, these
edges are transconductance values representing synaptic connections
between neurons.

Point types: Points can be grouped by types, which affect their behavior
and connectivity. Within the GT model framework, these types each have
their own temporal modulation constants (determining, e.g., whether the
cell bursts or fires singular spikes) and valence (excitatory or
inhibitory).

Global structure: Modelling the mammalian cortex, networks are assumed
to divide into a coarse-grained two-dimensional coordinate system of
layers (rows) and columns (columns). Each point is assigned to a
layer-column coordinate (called a "node"), having both local x-y
coordinates within that node and a global x-y coordinate within the
network.

Local structure: Each layer-column coordinate defines a "node"
containing a number of points determined by layer and type. Connections
(edges) within a node are determined by a local recurrence factor matrix
determining the transconductance between points of each type. These
edges are called "local".

Long-range projections: Connections (edges) between points in different
nodes are determined by a long-range projection motif and labelled with
the same of that motif.
