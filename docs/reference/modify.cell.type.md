# Modify existing cell type

This function modifies parameters of an existing cell type in the
current session. Parameters can be updated selectively. If a parameter
is not specified (or is specified as `NULL`), the existing value will be
kept.

## Usage

``` r
modify.cell.type(
  type_name,
  valence = NULL,
  tau_fast = NULL,
  tau_slow = NULL,
  tau_Vs = NULL,
  I_slow = NULL,
  U_Vs = NULL,
  max_spike_rate = NULL,
  transmission_velocity = NULL,
  spine_density = NULL,
  axon_target = NULL,
  I_spike = NULL,
  spike_potential = NULL,
  resting_potential = NULL,
  threshold = NULL,
  leak_conductance = NULL,
  axon_branch_count = NULL,
  dendrite_branch_count = NULL,
  branch_independence = NULL,
  branch_spread = NULL,
  apical_target_layer = NULL
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
  time to flow in).

- tau_slow:

  Time constant (ms) of the slow calcium (Ca2+) current (negative
  current, time to pump out).

- tau_Vs:

  Time constant (ms) for restoring presynaptic vesicles, i.e., recovery
  from short-term depression (STD).

- I_slow:

  Slow-current molecule (e.g., Ca2+) influx as concentration per spike
  (concentration/spike).

- U_Vs:

  Utilization ratio (concentration/spike) of vesicles per spike.

- max_spike_rate:

  Constant (spikes/ms) controlling estimation of spike rate and its
  maximum value.

- transmission_velocity:

  Transmission velocity (in microns/ms) along axon, for each neuron
  type.

- spine_density:

  Scale controlling percentage of dendrite nodes with spines: zero means
  none, one means all.

- axon_target:

  Character string giving target of axon projections, one of: "spine",
  "dendrite_shaft", "soma", or "axon_shaft".

- I_spike:

  Spike current, in pA; absolute value plus a little bit used as
  `dHdv_bound`.

- spike_potential:

  Peak potential during a spike, in mV.

- resting_potential:

  Resting potential, in mV; absolute value plus a little bit used as
  `v_bound`.

- threshold:

  Spike threshold, in mV.

- leak_conductance:

  Conductance controlling the leak current,
  `I_leak = leak_conductance * (resting_potential - v)`, in nS.

- axon_branch_count:

  Expected number of axon branches.

- dendrite_branch_count:

  Expected number of dendrite branches.

- branch_independence:

  Scale between 0 and 1; 0 = all branches connect to soma from single
  segment, 1 = all branches connect directly to soma.

- branch_spread:

  Scale between 0 and 1; 0 = no tendency to extend away from soma, 1 =
  straight line away from soma.

- apical_target_layer:

  Character string giving target layer for apical dendrites.

## Value

Nothing.
