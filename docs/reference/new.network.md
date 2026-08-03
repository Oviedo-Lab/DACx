# Initialize neuron network

This function initializes a new network object with specified
parameters. Networks are used to simulate two-dimensional cortical
patches (of layers and columns) using Growth Transform dynamical
systems.

## Usage

``` r
new.network()
```

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
