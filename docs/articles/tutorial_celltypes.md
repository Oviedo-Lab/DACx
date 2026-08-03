# Cell types

## Under construction!

This tutorial is under construction, but the scratch work below is
reasonably up-to-date as-of August 3, 2026.

``` r

# Clear the R workspace to start fresh
rm(list = ls())

# Set seed for reproducibility
set.seed(12345) 

# Load DACx package
library(DACx, quietly = TRUE) 
```

## Cell types

The three cell types (spiny stellates, PV, and SST) of the node are
pre-defined in the neurons package. To see all cell types known by the
package in the current session, we use the print.known.celltypes
function:

``` r

print.known.celltypes()
```

``` scroll-output
## Known cell types:
## 
## Type: SST
##   Valence: -1
##   Time constant, fast current (ms): 1
##   Time constant, slow current (ms): 60
##   STD recovery time constant (spikes/ms): 100
##   Slow current influx (concentration/spike): 0.035
##   Vesicle utilization ratio (concentration/spike): 0.05
##   Spike recovery rate (spikes/ms): 0.1
##   Leak conductance (nS): 10
##   Transmission velocity (micron/ms): 30000
##   Spine density: 0
##   Axon target: dendrite_shaft
##   Spike current (pA): 1000
##   Spike potential (mV): 35
##   Resting potential (mV): -70
##   Spike threshold (mV): -55
##   Axon branch count: 10
##   Dendrite branch count: 10
##   Branch independence: 0.75
##   Branch spread: 0.75
##   Apical target layer: none
## 
## Type: PV
##   Valence: -1
##   Time constant, fast current (ms): 1
##   Time constant, slow current (ms): 60
##   STD recovery time constant (spikes/ms): 100
##   Slow current influx (concentration/spike): 0.01
##   Vesicle utilization ratio (concentration/spike): 0.05
##   Spike recovery rate (spikes/ms): 0.1
##   Leak conductance (nS): 10
##   Transmission velocity (micron/ms): 30000
##   Spine density: 0
##   Axon target: soma
##   Spike current (pA): 1000
##   Spike potential (mV): 35
##   Resting potential (mV): -70
##   Spike threshold (mV): -55
##   Axon branch count: 10
##   Dendrite branch count: 10
##   Branch independence: 0.625
##   Branch spread: 0.625
##   Apical target layer: none
## 
## Type: callosal_PV
##   Valence: -1
##   Time constant, fast current (ms): 1
##   Time constant, slow current (ms): 60
##   STD recovery time constant (spikes/ms): 100
##   Slow current influx (concentration/spike): 0.01
##   Vesicle utilization ratio (concentration/spike): 0.05
##   Spike recovery rate (spikes/ms): 0.1
##   Leak conductance (nS): 10
##   Transmission velocity (micron/ms): 30000
##   Spine density: 0
##   Axon target: soma
##   Spike current (pA): 1000
##   Spike potential (mV): 35
##   Resting potential (mV): -70
##   Spike threshold (mV): -55
##   Axon branch count: 20
##   Dendrite branch count: 10
##   Branch independence: 0.25
##   Branch spread: 0.25
##   Apical target layer: none
## 
## Type: neurogliaform_cell
##   Valence: -1
##   Time constant, fast current (ms): 1
##   Time constant, slow current (ms): 120
##   STD recovery time constant (spikes/ms): 100
##   Slow current influx (concentration/spike): 0.035
##   Vesicle utilization ratio (concentration/spike): 0.05
##   Spike recovery rate (spikes/ms): 0.1
##   Leak conductance (nS): 10
##   Transmission velocity (micron/ms): 15000
##   Spine density: 0
##   Axon target: dendrite_shaft
##   Spike current (pA): 1000
##   Spike potential (mV): 35
##   Resting potential (mV): -70
##   Spike threshold (mV): -55
##   Axon branch count: 10
##   Dendrite branch count: 10
##   Branch independence: 0.75
##   Branch spread: 0.75
##   Apical target layer: none
## 
## Type: VIP
##   Valence: -1
##   Time constant, fast current (ms): 1
##   Time constant, slow current (ms): 60
##   STD recovery time constant (spikes/ms): 100
##   Slow current influx (concentration/spike): 0.035
##   Vesicle utilization ratio (concentration/spike): 0.05
##   Spike recovery rate (spikes/ms): 0.1
##   Leak conductance (nS): 10
##   Transmission velocity (micron/ms): 30000
##   Spine density: 0
##   Axon target: dendrite_shaft
##   Spike current (pA): 1000
##   Spike potential (mV): 35
##   Resting potential (mV): -70
##   Spike threshold (mV): -55
##   Axon branch count: 10
##   Dendrite branch count: 10
##   Branch independence: 0.625
##   Branch spread: 0.625
##   Apical target layer: none
## 
## Type: thalmacortical
##   Valence: 1
##   Time constant, fast current (ms): 1
##   Time constant, slow current (ms): 60
##   STD recovery time constant (spikes/ms): 150
##   Slow current influx (concentration/spike): 0.01
##   Vesicle utilization ratio (concentration/spike): 0.075
##   Spike recovery rate (spikes/ms): 0.1
##   Leak conductance (nS): 10
##   Transmission velocity (micron/ms): 30000
##   Spine density: 0.5
##   Axon target: spine
##   Spike current (pA): 1000
##   Spike potential (mV): 35
##   Resting potential (mV): -70
##   Spike threshold (mV): -55
##   Axon branch count: 5
##   Dendrite branch count: 10
##   Branch independence: 0.1
##   Branch spread: 0.9
##   Apical target layer: none
## 
## Type: spiny_stellate
##   Valence: 1
##   Time constant, fast current (ms): 1
##   Time constant, slow current (ms): 60
##   STD recovery time constant (spikes/ms): 100
##   Slow current influx (concentration/spike): 0.01
##   Vesicle utilization ratio (concentration/spike): 0.05
##   Spike recovery rate (spikes/ms): 0.1
##   Leak conductance (nS): 10
##   Transmission velocity (micron/ms): 30000
##   Spine density: 0.5
##   Axon target: spine
##   Spike current (pA): 1000
##   Spike potential (mV): 35
##   Resting potential (mV): -70
##   Spike threshold (mV): -55
##   Axon branch count: 10
##   Dendrite branch count: 10
##   Branch independence: 0.75
##   Branch spread: 0.75
##   Apical target layer: none
## 
## Type: pyramidal_L6
##   Valence: 1
##   Time constant, fast current (ms): 1
##   Time constant, slow current (ms): 60
##   STD recovery time constant (spikes/ms): 100
##   Slow current influx (concentration/spike): 0.01
##   Vesicle utilization ratio (concentration/spike): 0.05
##   Spike recovery rate (spikes/ms): 0.1
##   Leak conductance (nS): 10
##   Transmission velocity (micron/ms): 30000
##   Spine density: 0.5
##   Axon target: spine
##   Spike current (pA): 1000
##   Spike potential (mV): 35
##   Resting potential (mV): -70
##   Spike threshold (mV): -55
##   Axon branch count: 10
##   Dendrite branch count: 10
##   Branch independence: 0.25
##   Branch spread: 0.25
##   Apical target layer: L4
## 
## Type: callosal_pyramidal
##   Valence: 1
##   Time constant, fast current (ms): 1
##   Time constant, slow current (ms): 60
##   STD recovery time constant (spikes/ms): 100
##   Slow current influx (concentration/spike): 0.01
##   Vesicle utilization ratio (concentration/spike): 0.05
##   Spike recovery rate (spikes/ms): 0.1
##   Leak conductance (nS): 10
##   Transmission velocity (micron/ms): 30000
##   Spine density: 0.5
##   Axon target: spine
##   Spike current (pA): 1000
##   Spike potential (mV): 35
##   Resting potential (mV): -70
##   Spike threshold (mV): -55
##   Axon branch count: 20
##   Dendrite branch count: 10
##   Branch independence: 0.25
##   Branch spread: 0.25
##   Apical target layer: L1
## 
## Type: pyramidal
##   Valence: 1
##   Time constant, fast current (ms): 1
##   Time constant, slow current (ms): 60
##   STD recovery time constant (spikes/ms): 100
##   Slow current influx (concentration/spike): 0.01
##   Vesicle utilization ratio (concentration/spike): 0.05
##   Spike recovery rate (spikes/ms): 0.1
##   Leak conductance (nS): 10
##   Transmission velocity (micron/ms): 30000
##   Spine density: 0.5
##   Axon target: spine
##   Spike current (pA): 1000
##   Spike potential (mV): 35
##   Resting potential (mV): -70
##   Spike threshold (mV): -55
##   Axon branch count: 10
##   Dendrite branch count: 10
##   Branch independence: 0.25
##   Branch spread: 0.25
##   Apical target layer: L1
```

As can be seen from the output, cell types are defined by a number of
parameters.

### Cell type parameters

The parameters defining cell types in DACx are mostly biophysically
interpretable and can be set with, or tested against, experimental data.

#### Transmitter type

- *Valence:* 1 for excitatory cells, -1 for inhibitory cells. Code
  variable: valence.

#### Membrane kinetics

- *Fast current time constant:* Time constant (ms) of the fast sodium
  (Na+) current (positive current, time to flow in). Code variable:
  tau_fast.
- *Slow current time constant:* Time constant (ms) of the slow calcium
  (Ca2+) current (negative current, time to pump out). Code variable:
  tau_slow.
- *STD recovery time constant:* Time constant (ms) for restoring
  presynaptic vesicles, i.e., recovery from short-term depression (STD).
  Code variable: tau_Vs.
- *Slow current influx:* Slow-current molecule (e.g., Ca2+) influx as
  concentration per spike (concentration/spike). Code variable: I_slow.
- *Vesicle utilization ratio:* Utilization ratio (concentration/spike)
  of vesicles per spike. Code variable: U_Vs.
- *Spike recovery rate:* Constant (spikes/ms) controlling estimation of
  spike rate and its max value. Code variable: max_spike_rate.
- *Leak conductance:* Conductance (nS) controlling the leak current:
  I_leak = leak_conductance \* (resting_potential - v). Code variable:
  leak_conductance.

#### Intercell transmission

- *Transmission velocity:* The speed of the action potential between
  neurons (microns/ms). Code variable: transmission_velocity.
- *Spine density:* The proportion (a value between 0 and 1) of dendrite
  nodes expected to be spines. Code variable: spine_density.
- *Axon target:* The kind of nodes onto which the cell’s axons can
  synapse. Possible values include “spine”, “dendrite_shaft”, “soma”,
  and “axon_shaft”. Code variable axon_target.

#### Membrane potential and spiking

- *Spike current:* The peak current of the action potential (pA). Code
  variable: I_spike.
- *Spike potential:* The value of an action potential (mV). Code
  variable: spike_potential.
- *Resting potential:* The potential difference in electrical charge
  between the inside and outside of the cell at rest (mV). Code
  variable: resting_potential.
- *Threshold:* The potential difference in electrical charge between the
  inside and outside of the cell at which an action potential is
  triggered (mV). Code variable: threshold.

#### Neurite structure

- *Axon branch count:* An integer value giving the expected number axon
  branches. Code variable: axon_branch_count.
- *Dendrite branch count:* Same as axon branch count, but for dendrites.
  Code variable: dendrite_branch_count.
- *Branch independence:* The expected proportion (a value between 0 and
  1) of branches which connect directly to the soma, with 1 meaning all
  branches connect directly to the soma and zero meaning that all
  branches connect to the soma from a single segment. Code variable:
  branch_independence.
- *Branch spread:* Value between 0 and 1 controlling the tendency of
  arbor branches to repel away from the soma with 1 meaning a straight
  line away from the soma and 0 meaning no bias with respect to soma
  position. Code variable: branch_spread.

#### Apical dendrite parameters

- *Apical target layer:* Character string giving the name of the layer
  to which apical dendrite is expected to grow, if any; if none, “none”.
  Intended for modeling the circuit topology of pyramidal neurons. Code
  variable: apical_target_layer.

### Synaptic conductance

There is one important cell-type property not controlled directly
through the cell_type structure: synaptic conductance. Synaptic
conductance is the ease with which current flows into a synapse, and
thus represents synaptic “strength”. Of course, the strength of any one
synapse is determined by a history of activity – or, from a modeling
point of view, training. However, these values need to be initialized in
some way and we can expect certain combinations of cell type, layer, and
projection to be biased in certain directions for synaptic strength.
These initialization biases are controlled through arguments in the
set.network.structure function and the load.projection.into.motif
function. The former controls synaptic conductance biases within nodes
(local connections) and the latter controls synaptic conductance biases
between nodes (meso-scale connections). In both cases, the value is
specified in millisiemens (mS) and is set to 0.1nS by default, which is
equivalent to a 5 pico amp synaptic current at a 50mV potential.

### Modifying cell types

Cell types themselves are technically a C++ struc. Defined cell types
are stored in an unordered map in C++ that’s accessible via three R
wrapper functions. The function add.cell.type can be used to add a new
cell type to the map. The function modify.cell.type can be used to
modify an existing cell type. The function fetch.cell.type.params will
return the parameters of the named cell type as an R list.

### Principal neurons

The DACx package includes a special cell type, “principal”, which can be
called as well. This call doesn’t refer to any predefined cell type, but
instead triggers a layer-dependent lookup of known principal cell types.
The term “principal” simply means the primary cell type in a region. We
can see known principal cell types by calling the principal.neurons
function:

``` r

principal.neurons(print_nicely = TRUE)
```

``` scroll-output
## Principal neuron types by layer:
##   thalamus: thalmacortical
##   layer: spiny_stellate
##   L1: neurogliaform_cell
##   L2: pyramidal
##   L3: pyramidal
##   L4: spiny_stellate
##   L5: pyramidal
##   L6: pyramidal_L6
```

As can be seen, spiny stellates are the principal cell for layer 4. We
have used them for this demonstration because, unlike pyramidal cells,
they do not extend apical dendrites out of their home layer.
