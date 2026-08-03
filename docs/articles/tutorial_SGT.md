# Spatial growth-transform models

## Introduction

The simplest mathematical models of neural networks are built from
homogeneous McCulloch-Pitts neurons with atemporal, scalar-weight
connections between them. Biological neural networks, such as those in
the mammalian brain, are far more complex. They contain different types
of neurons each with their own electrical behavior and spatiotemporally
extended connections.

![Example diagram of a simple neural network from the original 1943
paper by McCulloch and Pitts](fig_MPN.png)

A sample neural network diagram from the original [1943
paper](https://link.springer.com/article/10.1007/BF02478259) by
McCulloch and Pitts.

Computational neuroscientists standardly model biological neural
networks as [dynamical systems](https://philpapers.org/rec/ELIMBM-3),
i.e., as having time-dependent states.
[Growth-transform](https://doi.org/10.3389/fnins.2020.00425) (GT)
models, extended to spiking neural networks by Gangopadhyay and
Chakrabartty, are one example. These models treat the change in membrane
potential over time, \partial v/\partial t, as a growth-transform of the
membrane potentials v on the net metabolic power \mathcal{H} of a
network, in the sense that it’s assumed that \partial v/\partial t
satisfies the [Baum-Eagon
inequality](https://doi.org/10.1090/S0002-9904-1967-11751-8):
\mathcal{H}(v)\leq \mathcal{H}(v + \partial v/\partial t) In other
words: \frac{\partial\mathcal{H}}{\partial v}\frac{\partial v}{\partial
t} \leq 0 Thus, GT models are essentially energy-minimization models of
neural dynamics. The model parameters provided by DACx are based on a
[biological
interpretation](https://Oviedo-Lab.org/DACx/articles/tutorial_math.md)
of the underlying mathematics of GT models.

In practical terms, instead of a one-dimensional weight between two
homogeneous neurons, GT models use both a transconductance parameter and
a temporal modulation factor to determine network behavior. The
transconductance parameter is the inverse of a traditional connection
weight, while the [temporal modulation
factor](https://Oviedo-Lab.org/DACx/articles/tutorial_membrane_temporal_dynamics.md)
allows for capturing the electrodynamics \partial v/\partial t of
different neuron types at a single spatial point.

The GT models implemented in the DACx package add a third aspect of
biological realism not captured by the original formulation of
Gangopadhyay and Chakrabartty: a transmission velocity parameter for
each neuron type. While the temporal modulation factor controls membrane
voltage over time at a *single* spatial point, the transmission velocity
parameter determines the rate at which changes in membrane voltage
*propagate between neurons*. For this reason, we refer to our GT models
as *spatial GT* (SGT) models.

SGT models thus allow for network topologies that not only capture
connection strengths between neurons, but also the different types of
neurons and their electrodynamics across both time *and space*. A
[separate
tutorial](https://Oviedo-Lab.org/DACx/articles/tutorial_network_topology.md)
demonstrates how to build spatially extended network topologies for
these models from circuit motifs. This tutorial explains SGT models in
more detail.

## Network nodes

Let’s set up the R environment by clearing the workspace, setting a
random-number generator seed, and loading the DACx package.

``` r

# Clear the R workspace to start fresh
rm(list = ls())

# Set seed for reproducibility
set.seed(12345) 

# Load DACx package
library(DACx, quietly = TRUE) 
```

Next, we create a new network object with the new.network function.

``` r

network.node <- new.network()
```

The object initialized by new.network is a minimal single-node network.
As we use the term, a “node” is not necessarily a single neuron, but is
rather a cluster of nearby neurons with local recurrent connections.
These nodes are expected to be (approximately) fully connected, with
cells of each type synapsing into cells of all other types. For this
tutorial, our nodes will include three distinct neuron types: an
excitatory type, layer 4 principal neurons (spiny stellates), and two
inhibitory types, parvalbumin (PV) and somatostatin (SST) interneurons.

Nodes are defined by their constitutive cell types and node size (i.e.,
expected number of neurons per type). Mimicking the structure of the
brain, nodes are arrayed into layers and columns, cortical and
subcortical regions, and hemispheres. The [network-topology
tutorial](https://Oviedo-Lab.org/DACx/articles/tutorial_network_topology.md)
explains how to set up this multi-node structure. For now, it suffices
to note that this structure is set with the set.network.structure
function. We will set up a single-node network, with an expected count
of 10 for the spiny stellates and 5 for each of the two inhibitory
interneuron types.

``` r

network.node <- set.network.structure(
    network.node,
    neuron_types     = c("spiny_stellate", "PV", "SST"),
    neurons_per_node = c(10, 5, 5)
  )
```

By default, the set.network.structure function creates no subcortical
layers and sets the number of cortical layers, columns, patches (a
secondary columnar axis perpendicular to the laminar axis), and
hemispheres each to one. That is, it makes a single node. It also
initializes local recurrent connections within that node. The node we
just created can be visualized with the plot.network function. To keep
the plot clean, we can use the arbor_density argument, which controls
the proportion of cells for which the generated arbors are shown.

``` r

plt <- plot.network(network.node, arbor_density = 0.1)
plt$plot
```

![](tutorial_SGT_files/figure-html/plot_cortical_patch_local_connections-1.png)

Here we see arbors for two cells, a spiny stellate and a PV interneuron.
These arbors are generated via a biased random walk, the biasing factors
fixed by cell type. The arbors include both axons and dendrites, which
can be visualized explicitly by changing the arbor coloring. Although
cells and arbors to-be-plotted are selected randomly, the plot.network
function returns masks for the cells and arbors plotted, and can take
this information to re-plot the same data again under different
settings, e.g., coloring by arbor type instead of by cell type:

``` r

plt <- plot.network(
    network.node, 
    arbor_density = 0.1,
    soma_mask     = plt$soma_mask,
    arbor_idx     = plt$arbor_idx,
    edge_color    = "is_axon"
  )
plt$plot
```

![](tutorial_SGT_files/figure-html/plot_cortical_patch_local_connections_axons-1.png)

The existence and number of connections between cells – synapses,
colored orange – are determined by the proximity of axons to dendrites.
After the arbors are created, a separate algorithm looks for axon nodes
within a certain small neighborhood of dendrite nodes and, if one is
found, extends the axon to connect with the dendrite.

As the axis labels indicate, SGT models assign to each neuron a spatial
coordinate giving its location along the laminar and columnar axes.
While the above are 2D plots, there is in fact a third dimension, the
“patch” dimension, which serves as a secondary columnar axis. All
coordinates are continuous and real-valued and are used in conjunction
with the transmission velocity parameter to simulate spike propagation
over the axonal arbors. Here, for example, we can plot a 3D
representation of our node including *all* arbors, colored by cell type:

``` r

plt <- plot.network(
    network.node, 
    arbor_density = 1.0,
    threedim      = TRUE
  )
plt$plot
```

As can be seen, even for a small single node, the arborization and
number of synapses can be extensive. We can quantify the extent by
calling the fetch.network.components function, which returns a list of
all components of the network, including a print out of summary data:

``` r

ntw <- fetch.network.components(network.node, include_arbors = TRUE)
```

``` scroll-output
## Summary of network:
##  Number of neurons: 16 
##  Number of synapses: 57 
##  Hemisphere names: left 
##  Number of hemispheres: 1 
##  Subortical layer names:  
##  Number of subcortical layers: 0 
##  Cortical layer names: layer 
##  Number of cortical layers: 1 
##  Number of columns: 1 
##  Number of patches: 1 
##  Cell types used: spiny_stellate, PV, SST 
##  Motifs used: local connections
```

As the axis labels in the previous plots also indicate, there is a
physically meaningful unit attached to the dimensions: microns. The
formulas used to compute spatial coordinates are discussed in the
[network-topology
tutorial](https://Oviedo-Lab.org/DACx/articles/tutorial_network_topology.md).

## SGT simulations

The function run.SGT runs a simulation of spiking activity across a
network using a SGT model. The function is just a wrapper over the SGT
method of C++ network objects. It takes four arguments:

1.  network: A network created by the new.network function and
    structured by the set.network.structure function.
2.  stimulus_current_matrix: A matrix of input currents (in mA) over the
    duration of the simulation, rows representing neurons and columns
    representing time bins.
3.  dt: Time-step size for simulation, in ms. Default is 10^{-3}.
4.  initial_potential: Initial value for membrane potential, applied to
    all cells. Default is -70 mV.

The number of columns of stimulus_current_matrix determines the length
of the simulation. In essence, the function run.SGT answers the
question: How would the network respond to this stimulus current over
this amount of time?

For example, let’s create a 500ms simulation for the node we created
above. From the above call to fetch.network.components, we know there
are 16 neurons in our network. We can load this value directly from the
function output:

``` r

n_neurons <- ntw$n_neurons
```

This gives us the number of rows needed in our stimulus current matrix.
For the number of columns, we need to know the number of time steps
required:

``` r

stim_time_ms <- 500
dt           <- 1e-3
n_steps      <- stim_time_ms/dt
cat("Number of time steps in the simulation:", n_steps)
```

``` scroll-output
## Number of time steps in the simulation: 5e+05
```

Now, suppose we want our simulation to involve a 200ms input current to
just the spiny stellates, starting at 100ms. We can compute the initial
and final time steps of this current, plus a mask for the spiny
stellates, as follows:

``` r

# Set stimulus start and length
stim_length_ms      <- 200
stim_start_ms       <- 100
# Find start and end steps of the input stimulus current
stim_length         <- stim_length_ms / dt
stim_start          <- stim_start_ms / dt 
stim_end            <- stim_start + stim_length - 1
# Find mask for principal neurons
spiny_stellate_mask <- ntw$neuron_type_name == "spiny_stellate"
```

A final question is how much current to apply. For this simulation,
we’ll use a constant current of 100 pico amp (100\times 10^{-9}mA) to
the spiny stellates during the stimulus period. It might be natural to
leave the input current at zero outside of the stimulus period, but
there is endogenous background activity even without exogenous input to
the network. So, we’ll specify a baseline input current of 10 pico amps
(10\times 10^{-9}mA) to all neurons throughout the entire stimulation
period:

``` r

pico_amp                   <- 1
stimulus_current_matrix    <- matrix(0, nrow = n_neurons, ncol = n_steps)
stimulus_current_matrix[spiny_stellate_mask, stim_start:stim_end] <- 400 * pico_amp
```

With the stimulus current matrix in hand, we can run the simulation:

``` r

sim_results <- run.SGT(
    network.node,
    stimulus_current_matrix,
    dt
  )
```

The result of the function run.SGT is a matrix of spike traces formatted
similar to stimulus_current_matrix: each row represents a neuron and
each column represents a time step from the simulation. Each entry is
the membrane potential of the neuron at that time bin, in mV. The order
of neurons and time steps matches across the input stimulus-current and
output spike-trace matrices, of course. In addition, a vector of spike
counts for each neuron (giving the number of times each neuron spiked)
in the network is also returned. Both are returned in a list of two
elements, sim_traces and spike_counts.

We can view the head of the simulation traces:

``` r

print(sim_results$sim_traces[1:10,1:10])
```

``` scroll-output
## NULL
```

As well as the head of the spike counts:

``` r

print(head(sim_results$spike_counts))
```

``` scroll-output
## [1] 6 6 6 6 6 6
```

The neurons package also includes the function plot.network.traces,
which takes a network object with a trace matrix and produces a plot of
the traces, putting all neurons of the same type together.

``` r

plot.network.traces(network.node)
```

![](tutorial_SGT_files/figure-html/unnamed-chunk-1-1.png)

We can manually add the start and end of the stimulus period to the plot
with vertical lines:

``` r

plt <- plot.network.traces(network.node, return_plot = TRUE)  +
  ggplot2::geom_vline(xintercept = stim_end * dt, linewidth = 1) +
  ggplot2::geom_vline(xintercept = stim_start * dt, linewidth = 1)
print(plt)
```

![](tutorial_SGT_files/figure-html/traces_with_stim-1.png)

## Spatial lag

What about I\_\mathrm{synaptic\\transmission}, the input current induced
by synaptic transmission across all synapses? We assume that the induced
post-synaptic current is equal to the inducing pre-synaptic potential,
modulated by the synaptic conduction. However, the complicating factor
is that, given spatial distance between cells and synaptic transmission
time, the relevant pre-synaptic potential v at time t from pre-synaptic
neuron N may not be v(t), but rather v(t^\prime) for some t^\prime \< t.

Let \vec{v} = \langle v_1, v_2, \ldots, v_n\rangle be the vector of
membrane potentials for all n neurons in the network at time t. Further,
let V be a n\times n time-dependent matrix which captures how each
neuron “sees” the others. Specifically, V\_{ij}(t) is the membrane
potential of neuron N_i that reaches neuron N_j at time t. Then,
assuming Q is an n\times n matrix of synaptic connections such that
Q\_{ij} is the conductance from neuron j to neuron i, we have that:
I\_\mathrm{synaptic\\transmission}(N_i) = \sum\_{j=1}^n
Q\_{ij}V\_{ji}(t)
