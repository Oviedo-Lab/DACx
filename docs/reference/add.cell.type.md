# Add new cell type

This function adds a user-defined cell type to the current session. It's
just a wrapper for the Rcpp-exported `add_cell_type` function.
Technically, `cell_type` is a struct defined in the Rcpp backend of the
DACx package. They are essentially labeled lists whose fields are
described by the parameters below. Each session stores cell types in the
Rcpp backend in an `unordered_map` with string labels. All parameters
come with biologically realistic (and mathematically workable) default
values, except for `type_name` and `valence`.

## Usage

``` r
add.cell.type(
  type_name,
  valence,
  tau_fast = 1,
  tau_slow = 60,
  tau_Vs = 100,
  I_slow = 0.01,
  U_Vs = 0.05,
  max_spike_rate = 0.1,
  transmission_velocity = 30000,
  spine_density = 0,
  axon_target = "dendrite_shaft",
  I_spike = 1000,
  spike_potential = 35,
  resting_potential = -70,
  threshold = -55,
  leak_conductance = 10,
  axon_branch_count = 10L,
  dendrite_branch_count = 10L,
  branch_independence = 0.5,
  branch_spread = 0.5,
  apical_target_layer = "none"
)
```

## Arguments

- type_name:

  Character string giving name of the cell type, e.g. "pyramidal", "PV",
  "SST", etc.

- valence:

  Valence of each neuron type, +1 for excitatory, -1 for inhibitory.

- tau_fast:

  Time constant (ms) of the fast sodium (Na+) current (positive current,
  time to flow in). Default is 1.0.

- tau_slow:

  Time constant (ms) of the slow calcium (Ca2+) current (negative
  current, time to pump out). Default is 60.0.

- tau_Vs:

  Time constant (ms) for restoring presynaptic vesicles, i.e., recovery
  from short-term depression (STD). Default is 100.0.

- I_slow:

  Slow-current molecule (e.g., Ca2+) influx as concentration per spike
  (concentration/spike). Default is 0.01.

- U_Vs:

  Utilization ratio (concentration/spike) of vesicles per spike. Default
  is 0.05.

- max_spike_rate:

  Constant (spikes/ms) controlling estimation of spike rate and its
  maximum value. Default is 0.1.

- transmission_velocity:

  Transmission velocity (in microns/ms) along axon, for each neuron
  type. Default value is 30e3.

- spine_density:

  Scale controlling percentage of dendrite nodes with spines: zero means
  none, one means all. Default is 0.0.

- axon_target:

  Character string giving target of axon projections, one of: "spine",
  "dendrite_shaft", "soma", or "axon_shaft". Default is
  "dendrite_shaft".

- I_spike:

  Spike current, in pA; absolute value plus a little bit used as
  `dHdv_bound`. Default value is 1e3 (i.e., 1 nA).

- spike_potential:

  Peak potential during a spike, in mV. Default value is 35.0.

- resting_potential:

  Resting potential, in mV; absolute value plus a little bit used as
  `v_bound`. Default value is -70.0.

- threshold:

  Spike threshold, in mV. Default value is -55.0.

- leak_conductance:

  Conductance controlling the leak current,
  `I_leak = leak_conductance * (resting_potential - v)`, in nS. Default
  value is 10.0.

- axon_branch_count:

  Expected number of axon branches. Default is 10.

- dendrite_branch_count:

  Expected number of dendrite branches. Default is 10.

- branch_independence:

  Scale between 0 and 1; 0 = all branches connect to soma from single
  segment, 1 = all branches connect directly to soma. Default is 0.5.

- branch_spread:

  Scale between 0 and 1; 0 = no tendency to extend away from soma, 1 =
  straight line away from soma. Default is 0.5.

- apical_target_layer:

  Character string giving target layer for apical dendrites. Default is
  "none".

## Value

Nothing.
