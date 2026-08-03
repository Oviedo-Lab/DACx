# Run Spatial Growth-Transform network simulation

This function uses a Spatial Growth-Transform (SGT) model to run a spike
simulation on a given network object for a specified matrix of membrane
currents over time. A matrix containing the spike traces of all neurons
over time after the simulation (neurons as rows, sample times as
columns) is saved in the network object, along with a vector of spike
counts for each neuron in the network. Both are returned on the R side
in a list.

## Usage

``` r
run.SGT(network, stimulus_current_matrix, dt = 0.001, initial_potential = -70)
```

## Arguments

- network:

  Network object on which to run the simulation.

- stimulus_current_matrix:

  Matrix of stimulus currents, with rows representing neurons and
  columns representing sample times.

- dt:

  Time step length in ms (default: 1e-3, i.e., 1 micosecond time steps).

- initial_potential:

  Initial value for membrane potential, applied to all cells (Default is
  -70 mV).

## Value

List containing the following elements:

- v_traces:

  Matrix of simulated spike traces for all neurons over time (neurons as
  rows, sample times as columns).

- spike_counts:

  Vector of spike counts for each neuron in the network.
