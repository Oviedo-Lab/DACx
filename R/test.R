
# Clear the R workspace to start fresh
rm(list = ls())

# Set seed for reproducibility
set.seed(12345)

# Load DACx package
library(DACx, quietly = TRUE)

network.node <- new.network()

modify.cell.type(
    "spiny_stellate",
    valence = NULL,
    temporal_modulation_bias = 2,
    temporal_modulation_timeconstant = NULL,
    temporal_modulation_amplitude = 0.0, # 0 = no bursting
    transmission_velocity = NULL,
    spine_density = NULL,
    axon_target = NULL,
    v_bound = 30,
    dHdv_bound = NULL,
    I_spike = NULL,
    spike_potential = NULL,
    resting_potential = 0,
    threshold = 1000,
    axon_branch_count = NULL,
    dendrite_branch_count = NULL,
    branch_independence = NULL,
    branch_spread = NULL,
    apical_target_layer = NULL
)

network.node <- set.network.structure(network.node, neurons_per_node = 1)

stim_time_ms <- 1000
dt <- 1e-3
n_steps <- stim_time_ms/dt

# Set stimulus start and length
stim_length_ms <- 20
stim_start_ms  <- 10
# Find start and end steps of the input stimulus current
stim_length    <- stim_length_ms / dt
stim_start     <- stim_start_ms / dt
stim_end       <- stim_start + stim_length - 1

pico_amp                   <- 1e-9 # 1e-9 is one pico amp, assuming units are mA
baseline_synaptic_current  <- 10 * pico_amp
expected_synapses_per_cell <- 10
baseline_active_synapses   <- 0.1 * expected_synapses_per_cell
rest_current               <- -baseline_active_synapses * baseline_synaptic_current
stimulus_current_matrix    <- matrix(rest_current, nrow = 1, ncol = n_steps)
#stimulus_current_matrix[spiny_stellate_mask, stim_start:stim_end] <- 0 # stimulus_current_matrix[spiny_stellate_mask, stim_start:stim_end] + 100 * 1e-9 # ten pico amps

sim_results <- run.SGT(
  network.node,
  stimulus_current_matrix,
  dt
)

plot.network.traces(network.node)
