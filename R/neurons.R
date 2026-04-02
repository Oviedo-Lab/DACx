
# Intro ################################################################################################################

# Neurons: A framework for neuron modelling, simulation, and analysis

# By Mike Barkasi
# GNU GPLv3: https://www.gnu.org/licenses/gpl-3.0.en.html
#   Copyright (c) 2025

#' @useDynLib neurons, .registration = TRUE
#' @import Rcpp
#' @import RcppEigen 
#' @import ggplot2
#' @import parallel
NULL

.onLoad <- function(libname, pkgname) {
    Rcpp::loadModule("neuron", TRUE)
    Rcpp::loadModule("motif", TRUE)
    Rcpp::loadModule("network", TRUE)
    Rcpp::loadModule("Projection", TRUE)
    init_known_celltypes()
  }

# Initialization for C++ object classes ################################################################################

#' Initialize neuron
#' 
#' This function initializes a new neuron object with specified parameters. The neuron objects are the main tools for running correlation analyses of spike data.
#' 
#' @param id_num Numeric identifier for the neuron (default: 0).
#' @param recording_name Recording (if any) on which this neuron is based (default: "not_provided").
#' @param type Modeled electrophysiology of neuron, e.g. "generic", "blackbox" "LIF", "McCullochPitts", "excitatory", "inhibitory", etc. (default: "generic").
#' @param genotype Genotype of animal, e.g. "WT", "KO", "MECP2", "transgenic", etc. (default: "not_provided").
#' @param sex Sex of animal (default: "not_provided").
#' @param hemi Hemisphere of neuron in which neuron is located, e.g. "left", "right" (default: "not_provided").
#' @param region Brain region in which neuron is located, e.g. "V1", "M1", "CA1", "PFC", etc. (default: "not_provided").
#' @param age Age of animal, e.g. "P0", "P7", "P14", "adult", etc. (default: "not_provided").
#' @param sim Logical indicating if neuron is simulated (TRUE) or recorded (FALSE) (default: FALSE).
#' @param unit_time Unit of time for spike raster or other recording data, e.g. "ms", "s", etc. (default: "ms").
#' @param unit_sample_rate Unit of sample rate for spike raster or other recording data, e.g. "Hz", "kHz", etc. (default: "Hz").
#' @param unit_data Unit of data for spike raster or other recording data, e.g. "mV", "spike", etc. (default: "mV").
#' @param t_per_bin Time (in above units) per bin, e.g., 1 ms per bin (default: 10.0).
#' @param sample_rate Sample rate (in above units), e.g., 1e4 Hz (default: 1e4).
#' @return A new neuron object.
#' @export
new.neuron <- function(
    id_num = 0, 
    recording_name = "not_provided", 
    type = "generic", 
    genotype = "not_provided",
    sex = "not_provided",
    hemi = "not_provided",
    region = "not_provided",
    age = "not_provided", 
    sim = FALSE, 
    unit_time = "ms", 
    unit_sample_rate = "Hz", 
    unit_data = "mV", 
    t_per_bin = 10.0, 
    sample_rate = 1e4
  ) {
    neuron <- new(
      neuron, 
      id_num, recording_name, type, genotype, sex, hemi, region, age, sim, unit_time, unit_sample_rate, unit_data, t_per_bin, sample_rate
    )
    return(neuron)
  }

#' Initialize network (circuit) motif
#' 
#' This function initializes a new motif object with specified parameters. Motifs are used for building networks of interconnected neurons. They are recipes for building internode projections within a neural network. They are "columnar", in the sense that they are repeated across cortical columns. 
#' 
#' @param motif_name Character string giving name of the motif (default: "not_provided").
#' @return A new motif object.
#' @export
new.motif <- function(
    motif_name = "not_provided"
  ) {
    motif <- new(
      motif,
      motif_name
    )
    return(motif)
  }

#' Initialize neuron network
#' 
#' This function initializes a new network object with specified parameters. Networks are used to simulate two-dimensional cortical patches (of layers and columns) using Growth Transform dynamical systems. 
#' 
#' Mathematically, networks are points (representing neurons) connected by directed edges. Within the growth-transform (GT) model framework, these edges are transconductance values representing synaptic connections between neurons.
#' 
#' Point types: Points can be grouped by types, which affect their behavior and connectivity. Within the GT model framework, these types each have their own temporal modulation constants (determining, e.g., whether the cell bursts or fires singular spikes) and valence (excitatory or inhibitory).
#' 
#' Global structure: Modelling the mammalian cortex, networks are assumed to divide into a coarse-grained two-dimensional coordinate system of layers (rows) and columns (columns). Each point is assigned to a layer-column coordinate (called a "node"), having both local x-y coordinates within that node and a global x-y coordinate within the network. 
#'  
#' Local structure: Each layer-column coordinate defines a "node" containing a number of points determined by layer and type. Connections (edges) within a node are determined by a local recurrence factor matrix determining the transconductance between points of each type. These edges are called "local". 
#' 
#' Long-range projections: Connections (edges) between points in different nodes are determined by a long-range projection motif and labelled with the same of that motif. 
#' 
#' @param network_name Character string giving name of the network (default: "not_provided").
#' @param recording_name Character string giving name of the recording on which this network is based (default: "not_provided").
#' @param type Character string giving type of network; "Growth_Transform" is the only option available (default: "Growth_Transform").
#' @param genotype Character string giving genotype of the animal from which the modelled network comes, e.g. "WT", "KO", "MECP2", "transgenic", etc. (default: "not_provided").
#' @param sex Character string giving sex of the animal from which the modelled network comes (default: "not_provided").
#' @param hemi Character string giving hemisphere of the animal from which the modelled network comes, e.g. "left", "right" (default: "not_provided").
#' @param region Character string giving brain region of the animal from which the modelled network comes, e.g. "V1", "M1", "CA1", "PFC", etc. (default: "not_provided").
#' @param age Character string giving age of the animal from which the modelled network comes, e.g. "P0", "P7", "P14", "adult", etc. (default: "not_provided").
#' @param unit_time Character string giving unit of time for spike raster or other recording data on which the model is based or being compared, e.g. "ms", "s", etc. (default: "ms").
#' @param unit_sample_rate Character string giving unit of sample rate for recording data on which the model is based or being compared, e.g. "Hz", "kHz", etc. (default: "Hz").
#' @param unit_potential Character string giving unit of cell-membrane potential for recording data on which the model is based or being compared, e.g. "mV", "uV", etc. (default: "mV").
#' @param unit_current Character string giving unit of cell current for recording data on which the model is based or being compared, e.g. "mA", "uA", etc. (default: "mA").
#' @param unit_conductance Character string giving unit of axon and dendrite conductance for recording data on which the model is based or being compared, e.g. "mS", "uS", etc. (default: "mS").
#' @param unit_distance Character string giving unit of distance axon and dendrite measurements on which the model is based or being compared, e.g. "micron", "mm", etc. (default: "micron").
#' @param t_per_bin Time (in above units) per bin, e.g., 1 ms per bin (default: 10.0).
#' @param sample_rate Sample rate (in above units), e.g., 1e4 Hz (default: 1e4).
#' @return A new network object.
#' @export
new.network <- function(
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
    t_per_bin = 1.0, 
    sample_rate = 1e4
  ) {
    network <- new(
      network, 
      network_name,
      recording_name,
      type,
      genotype,
      sex,
      hemi,
      region,
      age,
      unit_time,
      unit_sample_rate,
      unit_potential,
      unit_current,
      unit_conductance,
      unit_distance,
      t_per_bin,
      sample_rate
    )
    return(network)
  }

# Functions for network cell types #####################################################################################

#' Print known cell types 
#' 
#' This function prints names and all parameters for all cell types recognized in the current session. It's just a wrapper for the Rcpp-exported \code{print_known_celltypes} function. 
#' 
#' @rdname print-known-celltypes
#' @usage print.known.celltypes()
#' @return Nothing.
#' @export
print.known.celltypes <- function() print_known_celltypes()

#' Fetch cell type parameters 
#' 
#' This function returns the parameters for a named cell type in a list. It's just a wrapper for the Rcpp-exported \code{fetch_cell_type_params} function.
#' 
#' @return List of parameters for the named cell type. 
#' @export
fetch.cell.type.params <- function(type_name) fetch_cell_type_params(type_name)

#' Add new cell type
#' 
#' This function adds a user-defined cell type to the current session. It's just a wrapper for the Rcpp-exported \code{add_cell_type} function. Technically, \code{cell_type} is a \code{struc} defined in the Rcpp backend of the neurons package. They are essentially labeled lists with the following entries: \code{type_name}, \code{valence}, \code{temporal_modulation_bias}, \code{temporal_modulation_timeconstant}, \code{temporal_modulation_amplitude}, \code{transmission_velocity}, \code{v_bound}, \code{dHdv_bound}, \code{I_spike}, \code{coupling_scaling_factor}, \code{spike_potential}, \code{resting_potential}, and \code{threshold}. Each session stores cell types in the Rcpp backend in an \code{unordered_map} with \code{string} labels. All parameters come with biologically realistic (and mathematically workable) default values, except for \code{type_name} and \code{valence}. 
#' 
#' @param type_name Character string giving name of the cell type, e.g. "excitatory", "inhibitory", "PV", "SST", etc.
#' @param valence Valence of each neuron type, +1 for excitatory, -1 for inhibitory
#' @param temporal_modulation_bias Temporal modulation time (in ms) bias for each neuron type. Default value is 1e-3.
#' @param temporal_modulation_timeconstant Temporal modulation time (in ms) step for each neuron type. Default value is 1e0.
#' @param temporal_modulation_amplitude Temporal modulation time (in ms) cutoff for each neuron type. Default value is 5e-3.
#' @param transmission_velocity Transmission velocity (in microns/ms) for each neuron type. Default value is 30e3.
#' @param v_bound Potential bound, such that -v_bound <= v_traces <= v_bound, in unit_potential (mV), for each neuron in the network, based on its type. Default value is 85.0.
#' @param dHdv_bound Bound on derivative of metabolic energy wrt potential, such that dHdv_bound > abs(dHdv), in mA, for each neuron in the network, based on its type. Default value is 1.05e-6.
#' @param I_spike Spike current, in mA. Default value is 1e-6 (i.e., 1 nA).
#' @param coupling_scaling_factor Controls how energy used in synaptic transmission compares to that used in spiking. Default value is 1e-7, meaning that synaptic transmission uses 0.00001 percent of the energy used in spiking.
#' @param spike_potential Magnitude of each spike, in mV. Default value is 35.0.
#' @param resting_potential Resting potential, in mV. Default value is -70.0.
#' @param threshold Spike threshold, in mV. Default value is -55.0.
#' @param process_node_count Expected number of process nodes over the length of one process branch. Default is 10.
#' @param axon_branch_count Expected number of axon branches. Default is 10. 
#' @param dendrite_branch_count Expected number of dendrite branches. Default is 10. 
#' @return Nothing.
#' @export
add.cell.type <- function(
    type_name,
    valence,
    temporal_modulation_bias = 1e-3,
    temporal_modulation_timeconstant = 1e0,
    temporal_modulation_amplitude = 5e-3,
    transmission_velocity = 30e3,
    v_bound = 85.0,
    dHdv_bound = 1.05e-6,
    I_spike = 1e-6,
    coupling_scaling_factor = 1e-7,
    spike_potential = 35.0,
    resting_potential = -70.0,
    threshold = -55.0,
    process_node_count = 10,
    axon_branch_count = 10,
    dendrite_branch_count = 10
  ) {
    add_cell_type(
      type_name, 
      valence, 
      temporal_modulation_bias, 
      temporal_modulation_timeconstant, 
      temporal_modulation_amplitude, 
      transmission_velocity, 
      v_bound, 
      dHdv_bound, 
      I_spike, 
      coupling_scaling_factor, 
      spike_potential, 
      resting_potential, 
      threshold,
      process_node_count,
      axon_branch_count,
      dendrite_branch_count
    )
  }

#' Modify existing cell type 
#' 
#' This function modifies parameters of an existing cell type in the current session. Parameters can be updated selectively. If the parameter is not specified at all or is specified as \code{NULL}, the existing parameter will be left in place. 
#' 
#' @param type_name Character string giving name of the cell type, e.g. "excitatory", "inhibitory", "PV", "SST", etc.
#' @param valence Valence of each neuron type, +1 for excitatory, -1 for inhibitory
#' @param temporal_modulation_bias Temporal modulation time (in ms) bias for each neuron type
#' @param temporal_modulation_timeconstant Temporal modulation time (in ms) step for each neuron type
#' @param temporal_modulation_amplitude Temporal modulation time (in ms) cutoff for each neuron type
#' @param transmission_velocity Transmission velocity (in microns/ms) for each neuron type
#' @param v_bound Potential bound, such that -v_bound <= v_traces <= v_bound, in mV, for each neuron in the network, based on its type
#' @param dHdv_bound Bound on derivative of metabolic energy wrt potential, such that dHdv_bound > abs(dHdv), in mA, for each neuron in the network, based on its type
#' @param I_spike Spike current, in mA
#' @param coupling_scaling_factor Controls how energy used in synaptic transmission compares to that used in spiking
#' @param spike_potential Magnitude of each spike, in mV
#' @param resting_potential Resting potential, in mV
#' @param threshold Spike threshold, in mV
#' @param process_node_count Expected number of process nodes over the length of one process branch
#' @param axon_branch_count Expected number of axon branches
#' @param dendrite_branch_count Expected number of dendrite branches
#' @return Nothing.
#' @export
modify.cell.type <- function(
    type_name,
    valence = NULL,
    temporal_modulation_bias = NULL,
    temporal_modulation_timeconstant = NULL,
    temporal_modulation_amplitude = NULL,
    transmission_velocity = NULL,
    v_bound = NULL,
    dHdv_bound = NULL,
    I_spike = NULL,
    coupling_scaling_factor = NULL,
    spike_potential = NULL,
    resting_potential = NULL,
    threshold = NULL,
    process_node_count = NULL,
    axon_branch_count = NULL,
    dendrite_branch_count = NULL
  ) {
    existing_params <- fetch.cell.type.params(type_name)
    if (is.null(valence)) valence <- existing_params$valence
    if (is.null(temporal_modulation_bias)) temporal_modulation_bias <- existing_params$temporal_modulation_bias
    if (is.null(temporal_modulation_timeconstant)) temporal_modulation_timeconstant <- existing_params$temporal_modulation_timeconstant
    if (is.null(temporal_modulation_amplitude)) temporal_modulation_amplitude <- existing_params$temporal_modulation_amplitude
    if (is.null(transmission_velocity)) transmission_velocity <- existing_params$transmission_velocity
    if (is.null(v_bound)) v_bound <- existing_params$v_bound
    if (is.null(dHdv_bound)) dHdv_bound <- existing_params$dHdv_bound
    if (is.null(I_spike)) I_spike <- existing_params$I_spike
    if (is.null(coupling_scaling_factor)) coupling_scaling_factor <- existing_params$coupling_scaling_factor
    if (is.null(spike_potential)) spike_potential <- existing_params$spike_potential
    if (is.null(resting_potential)) resting_potential <- existing_params$resting_potential
    if (is.null(threshold)) threshold <- existing_params$threshold
    if (is.null(process_node_count)) process_node_count <- existing_params$process_node_count
    if (is.null(axon_branch_count)) axon_branch_count <- existing_params$axon_branch_count
    if (is.null(dendrite_branch_count)) dendrite_branch_count <- existing_params$dendrite_branch_count
    modify_cell_type(
      type_name, 
      valence, 
      temporal_modulation_bias, 
      temporal_modulation_timeconstant, 
      temporal_modulation_amplitude, 
      transmission_velocity, 
      v_bound, 
      dHdv_bound, 
      I_spike, 
      coupling_scaling_factor, 
      spike_potential, 
      resting_potential, 
      threshold,
      process_node_count,
      axon_branch_count,
      dendrite_branch_count
    )
  }

# Functions for network ################################################################################################

#' Load projection into motif
#' 
#' This function loads a projection schema into a motif object. Projections define internode connectivity within a network built using the motif.
#' 
#' @param motif Motif object into which to load the projection.
#' @param presynaptic_layer Character string giving layer of presynaptic neuron, e.g. "L2/3", "L4", "L5", "L6", etc.
#' @param postsynaptic_layer Character string, or vector of character strings, giving layer of postsynaptic neuron, e.g. "L2/3", "L4", "L5", "L6", etc.
#' @param density Numeric giving density of the projection; if left NULL, will use presynaptic_density and postsynaptic_density; if set, will use this value for both presynaptic_density and postsynaptic_density.
#' @param presynaptic_density Numeric giving density of presynaptic neuron type in presynaptic layer (e.g., ratio of neurons per node, default: 0.5).
#' @param postsynaptic_density Numeric giving density of postsynaptic neuron type in postsynaptic layer (e.g., ratio of neurons per node, default: 0.5).
#' @param presynaptic_type Character string giving type of presynaptic neuron, e.g. "excitatory", "inhibitory", etc. (default: "principal").
#' @param postsynaptic_type Character string giving type of postsynaptic neuron, e.g. "excitatory", "inhibitory", etc. (default: "principal").
#' @param max_col_shift_up Maximum number of columns upwards (increasing columnar indexes) that the projection can reach (default: 0, should be positive integer).
#' @param max_col_shift_down Maximum number of columns downwards (decreasing columnar indexes) that the projection can reach (default: 0, should be positive integer).
#' @param connection_strength Numeric giving overall strength of the projection (default: 1.0).
#' @return The updated motif object with the new projection loaded.
#' @export
load.projection.into.motif <- function(
    motif,
    presynaptic_layer,
    postsynaptic_layer,
    density = NULL,
    presynaptic_density = 0.5,
    postsynaptic_density = 0.5,
    presynaptic_type = "principal",
    postsynaptic_type = "principal",
    max_col_shift_up = 0,
    max_col_shift_down = 0,
    connection_strength = 1.0
  ) {
    if (length(presynaptic_layer) != 1) {
      stop("presynaptic_layer must be a single layer name.")
    }
    # ... for each target layer
    for (psl in postsynaptic_layer) {
      # Initialize new projection object
      proj <- new(Projection)
      # Check density 
      if (!is.null(density)) {
        presynaptic_density <- density
        postsynaptic_density <- density
      }
      # Load projection parameters 
      proj$pre_type <- presynaptic_type
      proj$pre_layer <- presynaptic_layer
      proj$pre_density <- max(min(presynaptic_density, 1), 0)
      proj$post_type <- postsynaptic_type
      proj$post_layer <- psl
      proj$post_density <- max(min(postsynaptic_density, 1), 0)
      # Add projection to motif
      motif$load_projection(
        proj,
        as.integer(max_col_shift_up),
        as.integer(max_col_shift_down),
        connection_strength
      )
    }
    return(motif)
  }

#' Set network structure
#' 
#' This function sets the structure of a network object, defining its layers, columns, neuron types, and local connectivity parameters. It also generates local nodes based on the specified structure.
#' 
#' @param network Network object to configure.
#' @param neuron_types Character vector giving types of neurons in the network, e.g. c("principal", "interneuron").
#' @param neuron_type_valences Numeric vector giving valences of each neuron type, e.g. c(1, -1) for excitatory and inhibitory neurons.
#' @param neuron_type_temporal_modulation Numeric matrix giving temporal modulation time components (for modulation time in the unit_time of the network) for each neuron type: bias, step size, and count cutoff (rows as neuron types, columns as components). Will example a single value or a vector of length three. 
#' @param layer_names Character vector giving names of layers in the network, e.g. c("L2/3", "L4", "L5", "L6").
#' @param n_layers Integer giving number of layers in the network.
#' @param n_columns Integer giving number of columns in the network.
#' @param patch_depth Integer giving the number of "patches" (n_layers x n_columns sheets) in the network.
#' @param layer_height Numeric giving height of each layer (in units specified at network creation, default unit is microns, default value is 250.0).
#' @param column_diameter Numeric giving diameter of each column (in units specified at network creation, default unit is microns, default value is 130.0).
#' @param layer_separation_factor Numeric giving mean distance between layers as a fraction of layer height (default: 3.0).
#' @param column_separation_factor Numeric giving mean distance between columns as a fraction of column diameter (default: 3.5).
#' @param patch_separation_factor Numeric giving mean distance between network patches as a fraction of column diameter (default: 3.5). 
#' @param neurons_per_node Matrix giving number of neurons of each type per node in each layer; dimensions must match n_layers (rows) and length of neuron_types (columns).
#' @param recurrence_factors List of matrices giving local recurrence factors for each layer; each matrix must have dimensions matching length of neuron_types (rows and columns).
#' @param pruning_threshold_factor Numeric giving factor for pruning weak connections within nodes; connections with strength below this factor times the maximum connection strength in the node will be pruned (default: 0.1).
#' @return The updated network object with the specified structure and local nodes generated.
#' @export
set.network.structure <- function(
    network,
    neuron_types = c("principal"),
    layer_names = c("layer"),
    n_layers = 1,
    n_columns = 1,
    patch_depth = 1,
    layer_height = 250.0,
    column_diameter = 130.0,
    layer_separation_factor = 3.0,
    column_separation_factor = 3.5,
    patch_separation_factor = 3.5,
    neurons_per_node = 30,
    recurrence_factors = 0.5,
    pruning_threshold_factor = 0.1
  ) {
    # Run checks 
    n_neuron_types <- length(neuron_types)
    if (length(layer_names) != n_layers) {
      if (n_layers > length(layer_names) && length(layer_names) == 1) {
        layer_names <- paste0(layer_names, "_", seq_len(n_layers))
      } else if (n_layers < length(layer_names) && n_layers == 1) {
        n_layers <- length(layer_names)
      } else {
        stop("Length of layer_names does not match n_layers, and neither is inferable from the other.")
      }
    }
    if (!is.null(dim(neurons_per_node))) {
      npn_dim <- dim(neurons_per_node)
    } else {
      if (length(neurons_per_node) == 1) {
        if (n_layers > 1) {
          neurons_per_node <- matrix(neurons_per_node, nrow = n_layers, ncol = n_neuron_types)
          npn_dim <- dim(neurons_per_node)
        } else {
          neurons_per_node <- matrix(rep(neurons_per_node, n_neuron_types), nrow = 1, ncol = n_neuron_types)
          npn_dim <- c(1, length(neurons_per_node))
        }
      } else if (length(neurons_per_node) == n_neuron_types) {
        neurons_per_node <- matrix(rep(neurons_per_node, n_layers), nrow = n_layers, ncol = n_neuron_types, byrow = TRUE)
        npn_dim <- dim(neurons_per_node)
      } else {
        stop("Dimensions of neurons_per_node must match n_layers and length of neuron_types, or be inferable from them.")
      }
    }
    if (any(npn_dim != c(n_layers, n_neuron_types))) {
      stop("Dimensions of neurons_per_node must match n_layers and length of neuron_types.")
    }
    if (!("list" %in% class(recurrence_factors))) {
      if ("matrix" %in% class(recurrence_factors) || "numeric" %in% class(recurrence_factors)) {
        recurrence_factors_matrix <- as.matrix(recurrence_factors)
        if (length(recurrence_factors_matrix) != n_neuron_types^2) {
          if (length(recurrence_factors_matrix) == 1) {
            recurrence_factors_matrix <- matrix(
              recurrence_factors_matrix, 
              nrow = n_neuron_types, 
              ncol = n_neuron_types
            )
          } else {
            stop("Dimensions of recurrence_factors matrix must match length of neuron_types, or be a single scalar.")
          }
        }
        recurrence_factors <- list()
        for (l in seq_len(n_layers)) recurrence_factors[[l]] <- recurrence_factors_matrix
      } else {
        stop("recurrence_factors must be a list of matrices or a single matrix.")
      }
    } else if (length(recurrence_factors) != n_layers) {
      stop("Length of recurrence_factors list must match n_layers.") 
    } else {
      for (l in seq_len(n_layers)) {
        rf_dim <- dim(recurrence_factors[[l]])
        if (length(rf_dim) != 2) {
          stop(paste0("recurrence_factors[[", l, "]] must be a matrix."))
        }
        if (any(rf_dim != c(n_neuron_types, n_neuron_types))) {
          stop(paste0("Dimensions of recurrence_factors[[", l, "]] must match length of neuron_types."))
        }
      }
    }
    # Set structure
    network$set_network_structure(
      neuron_types,
      layer_names,
      as.integer(n_layers),
      as.integer(n_columns),
      as.integer(patch_depth),
      layer_height,
      column_diameter,
      layer_separation_factor,
      column_separation_factor,
      patch_separation_factor,
      neurons_per_node,
      recurrence_factors,
      pruning_threshold_factor
    )
    # Make local nodes and return
    network$make_local_nodes()
    return(network)
  }

#' Apply circuit motif to network
#' 
#' This function applies a circuit motif to a network object, adding long-range projections between nodes in the network based on the motif's defined projections.
#' 
#' @param network Network object to which the motif will be applied.
#' @param motif Motif object defining the circuit motif to apply.
#' @return The updated network object with the motif applied.
#' @export
apply.circuit.motif <- function(
    network,
    motif
  ) {
    network$apply_circuit_motif(motif)
    return(network)
  }

#' Plot network as directed graph
#' 
#' This function plots a network object as a directed graph using ggplot2. Nodes represent neurons, and directed edges represent connections between them. The plot can be customized by selecting which motif to display and how to color the edges.
#' 
#' @name plot.network
#' @rdname plot-network
#' @usage plot.network(
#'  network, 
#'  title = NULL, 
#'  plot_motif = "local connections", 
#'  edge_color = "pre_type", 
#'  cell_color = "layer", 
#'  cell_size_factor = 5.0, 
#'  arrow_size_factor = 0.5, 
#'  return_plot = FALSE
#' )
#' @param network Network object to plot.
#' @param title Title for the plot (default: "Cortical Patch" or network name (if provided), plus plot motif name(s)).
#' @param plot_motif Character string specifying which motif to plot; options include "local" for local connections within each node or the name of a long-range projection motif (default: "local connections").
#' @param edge_color Character string specifying how to color the edges; options include "pre_type" to color by presynaptic neuron type, "post_type" to color by postsynaptic neuron type, or "motif" to color by motif type (default: "pre_type").
#' @param cell_color Character string specifying how to color the nodes; options include "layer" to color by layer index or "type" to color by neuron type (default: "layer").
#' @param cell_size_factor Numeric value controlling how cell size in the plot scales to the number of cells. 
#' @param arrow_size Numeric value indicating the size of the arrows representing edges (default: 0.05).
#' @param return_plot Logical indicating whether to return the ggplot object (TRUE) or print the plot directly (FALSE) (default: FALSE).
#' @return Either prints the plot directly or returns the ggplot object, depending on the value of return_plot.
#' @export
plot.network <- function(
    network,
    title = NULL,
    plot_motif = "local connections",
    edge_color = "pre_type",
    cell_color = "layer",
    cell_size_factor = 5.0,
    arrow_size_factor = 0.5,
    return_plot = FALSE
  ) {
    
    # Get network components
    ntw <- network$fetch_network_components(FALSE) # Retrieve arbors? 
   
    # Set plot title 
    if (is.null(title)) {
      if (ntw$network_name == "not_provided") {
        title <- "Cortical Patch"
      } else {
        title <- ntw$network_name
      }
      title <- paste0(title, ", ", paste0(plot_motif, collapse = ", "))
    }
    
    # Get unit information
    network_units <- ntw$units
    
    # Get cell coordinates and types 
    neuron_coordinates <- ntw$coordinates_spatial
    neuron_types <- ntw$neuron_type_name
    
    # Get layer information 
    layer_names <- ntw$layer_names
    neuron_layer <- as.factor(layer_names[ntw$coordinates_node[,"layer_idx"]])
    
    # Get cell edge pairs
    edges <- matrix(0, nrow = 0, ncol = 5)
    edge_type_names <- ntw$edge_type_names
    edge_type_mask <- edge_type_names %in% plot_motif
    edge_type_names <- edge_type_names[edge_type_mask]
    n_edge_types <- length(edge_type_names)
    et_masked <- which(edge_type_mask)
    for (et in seq_along(edge_type_names)) {
      et_name <- edge_type_names[et]
      et_edges <- ntw$edge_idx_by_type[[et_masked[et]]]
      et_edges <- cbind(
        et_edges, 
        rep(et_name, nrow(et_edges)),
        neuron_types[et_edges[,"pre_neuron_idx"]],
        neuron_types[et_edges[,"post_neuron_idx"]]
      )
      edges <- rbind(edges, et_edges)
    }
    edges <- as.data.frame(edges)
    colnames(edges) <- c("pre_idx", "post_idx", "motif", "pre_type", "post_type")
    
    # Create cells dataframe
    cells <- data.frame(
      idx = c(1:nrow(neuron_coordinates)), 
      x = neuron_coordinates[,"x"], 
      y = neuron_coordinates[,"y"],
      layer = neuron_layer,
      type = neuron_types
    )
    
    # Find coordinates for start and end of edges
    edges$x_start <- cells[edges$pre_idx, "x"]
    edges$y_start <- cells[edges$pre_idx, "y"]
    edges$x_end <- cells[edges$post_idx, "x"]
    edges$y_end <- cells[edges$post_idx, "y"]
    
    # Set point size to scale with number of cells
    n_cells <- nrow(cells)
    cell_size <- cell_size_factor * 100 / n_cells
    
    # Set arrow size to scale with number of edges
    n_edges <- nrow(edges)
    arrow_size <- arrow_size_factor / n_edges
    
    # Scake alpha by number of edges 
    edge_alpha <- max(0.1, min(1, n_cells / (n_edges + 1)))
    
    # Make colors 
    if (length(unique(as.character(cells[,cell_color]))) == 1) cells[,cell_color] <- "cell"
    colored_labels <- unique(
      c(unique(as.character(edges[,edge_color])), 
        unique(as.character(cells[,cell_color])))
      )
    known_label_colors <- list(
      "cell" = "gray50",
      "L1" = "gray50",
      "L2" = "lightskyblue3",
      "L2/3" = "lightskyblue2",
      "L23" = "lightskyblue2",
      "L3" = "lightskyblue1",
      "L4" = "slateblue1",
      "L5" = "skyblue1",
      "L6" = "royalblue1",
      "principal" = "green3",
      "PN" = "green3", 
      "excitatory" = "green3",
      "interneuron" = "red",
      "inhibitory" = "red", 
      "PV" = "darkred",
      "SOM" = "darkorchid",
      "SST" = "darkorchid",
      "VIP" = "darkorange"
    )
    unknown_label_colors <- c("aquamarine1", "gray95", "gray55", "gray75", "cyan", "cornflowerblue", "coral", "burlywood", "darkolivegreen")
    label_colors <- rep("white", length(colored_labels))
    names(label_colors) <- colored_labels
    for (cl in seq_along(colored_labels)) {
      label <- colored_labels[cl]
      hit_mask <- grepl(label, names(known_label_colors))
      if (any(hit_mask)) {
        hit_idx <- which(hit_mask)[1]
        label_colors[cl] <- known_label_colors[[hit_idx]]
      } else {
        label_colors[cl] <- sample(unknown_label_colors, 1)
      }
    }
    
    # Plot
    title_size <- 14 
    axis_size <- 12 
    legend_size <- 10
    plt <- ggplot2::ggplot() +
      # cells as points
      ggplot2::geom_point(data = cells, size = cell_size, ggplot2::aes(x = x, y = y, color = .data[[cell_color]])) +
      # edges as arrows
      ggplot2::geom_segment(
        data = edges,
        ggplot2::aes(x = x_start, y = y_start, xend = x_end, yend = y_end, color = .data[[edge_color]]),
        arrow = ggplot2::arrow(length = ggplot2::unit(arrow_size, "npc"), type = "closed"),
        alpha = edge_alpha
      ) +
      ggplot2::theme_minimal() +
      ggplot2::labs(
        title = title, 
        x = paste0("columnar coordinate (", network_units$distance, ")"), 
        y = paste0("laminar coordinate (", network_units$distance, ")")
        ) + 
      ggplot2::scale_colour_manual(
        name = "Types",
        values = label_colors
      ) +
      ggplot2::guides(color = ggplot2::guide_legend(override.aes = list(alpha = 1))) +
      ggplot2::theme(
        panel.background = ggplot2::element_rect(fill = "white", colour = NA),
        plot.background  = ggplot2::element_rect(fill = "white", colour = NA),
        #panel.grid = ggplot2::element_line(color = "gray80", linewidth = 0.25),
        plot.title = ggplot2::element_text(hjust = 0.5, size = title_size),
        axis.title = ggplot2::element_text(size = axis_size),
        axis.text = ggplot2::element_text(size = axis_size),
        legend.title = ggplot2::element_text(size = legend_size),
        legend.text = ggplot2::element_text(size = legend_size) #,
        #legend.position = "bottom"
      )
    
    if (return_plot) {
      return(plt)
    } else {
      print(plt)
      return(invisible(NULL))
    }
    
  }

#' Plot spike traces for network from SGT simulation 
#' 
#' This function plots spike traces for a network object from a Spatial Growth-Transform (SGT) simulation. 
#' 
#' @name plot.network.traces
#' @rdname plot-network-traces
#' @usage plot.network.traces(network, return_plot)
#' @param network Network object with SGT simulation traces to plot.
#' @param return_plot Logical indicating whether to return the ggplot object (TRUE) or print it (FALSE) (default: FALSE).
#' @return A ggplot object showing spike traces for all neurons in the network over time.
#' @export
plot.network.traces <- function(
    network,
    return_plot = FALSE
  ) {
    
    # Get the traces to print
    sim_traces <- network$fetch_sim_traces_R()
    
    # Get network components
    ntw <- network$fetch_network_components(FALSE) # Retrieve arbors?
    
    # Initialize R data frame for ggplot
    sim_traces_long <- data.frame()
    time_seq <- seq(1, by = ntw$sim_dt, length.out = ncol(sim_traces))
    sim_steps <- c(1:ncol(sim_traces))
    for (i in 1:nrow(sim_traces)) {
      neuron_trace <- data.frame(
        time = time_seq,
        potential = sim_traces[i, sim_steps],
        id = i,
        type = ntw$neuron_type_name[i]
      )
      sim_traces_long <- rbind(sim_traces_long, neuron_trace)
    }
    sim_traces_long$id <- as.character(sim_traces_long$id)
    
    # Make plot
    title_size <- 14 
    axis_size <- 12 
    legend_size <- 10
    plt <- ggplot2::ggplot(sim_traces_long, ggplot2::aes(x = time, y = potential, group = id, color=id)) +
      ggplot2::geom_line() +
      ggplot2::facet_wrap(~ type, ncol = 1) +
      ggplot2::theme_minimal() +
      ggplot2::theme(
        panel.background = ggplot2::element_rect(fill = "white", colour = NA),
        plot.background  = ggplot2::element_rect(fill = "white", colour = NA),
        plot.title = ggplot2::element_text(hjust = 0.5, size = title_size),
        axis.title = ggplot2::element_text(size = axis_size),
        axis.text = ggplot2::element_text(size = axis_size),
        legend.title = ggplot2::element_text(size = legend_size),
        legend.text = ggplot2::element_text(size = legend_size),
        legend.position = "none") +
      ggplot2::labs(
        title = "SGT Simulation Traces",
        x = paste0("Time (", ntw$units$time, ")"),
        y = paste0("Membrane Potential (", ntw$units$potential, ")")
      )
    
    if (return_plot) {
      return(plt)
    } else {
      print(plt)
      return(invisible(NULL))
    }
    
  }

#' Run Spatial Growth-Transform network simulation
#' 
#' This function uses a Spatial Growth-Transform (SGT) model to run a spike simulation on a given network object for a specified matrix of input currents over time. A matrix containing the spike traces of all neurons over time after the simulation (neurons as rows, sample times as columns) is saved in the network object, along with a vector of spike counts for each neuron in the network. Both are returned on the R side in a list.
#' 
#' @param network Network object on which to run the simulation.
#' @param stimulus_current_matrix Matrix of input currents, with rows representing neurons and columns representing sample times.
#' @param dt Time step length in the unit_time of the network (default: 1e-3, or 1 micosecond time steps).
#' @return List containing the following elements: \item{sim_traces}{Matrix of simulated spike traces for all neurons over time (neurons as rows, sample times as columns).} \item{spike_counts}{Vector of spike counts for each neuron in the network.} 
#' @export
run.SGT <- function(
    network,
    stimulus_current_matrix,    # matrix of input currents (rows: neurons, columns: time bins)
    dt = 1e-3                   # time step length, in ms
  ) {
    network$SGT(
      stimulus_current_matrix,
      dt
    )
    sim_traces <- network$fetch_sim_traces_R()
    spike_counts <- network$fetch_spike_counts_R()
    return(list(sim_traces = sim_traces, spike_counts = spike_counts))
  }

# Functions for neuron analysis ########################################################################################

#' Load neuron data from raster file
#' 
#' This function loads spike raster data from a data frame or CSV file and converts each unique cell into a neuron object. The raster data frame must contain columns for cell identifier, spike time in milliseconds, and trial number. Optional metadata columns can also be included.
#' 
#' @param raster_df Data frame (or file name to csv importable as such), each row a spike; must have columns: cell, time_in_ms, trial; optional columns: recording_name, hemisphere, genotype, sex, region, age.
#' @param bin_size Size of time bins in milliseconds (default: 10).
#' @param min_duration Minimum trial duration in milliseconds (default: 0).
#' @param max_displacement Max displacement of trial center in milliseconds (default: 1e9, i.e., "infinity").
#' @param sample_rt Sample rate in the default unit for neuron objects, Hz (default: 1e4).
#' @param time_cutoff Maximum time (in ms) to include spikes; spikes occurring after this time will be excluded (default: Inf).
#' @return A list of neuron objects, one per unique cell in the raster data.
#' @export
load.rasters.as.neurons <- function(
    raster_df,
    bin_size = 10.0, 
    min_duration = 0,
    max_displacement = 1e9,
    sample_rt = 1e4,
    time_cutoff = Inf
  ) {
    
    # Load data
    if (class(raster_df) == "character") {
      raster_df <- as.data.frame(read.csv(raster_df))
    }
    
    # Initiate list to hold neurons
    neuron_list <- list()
    
    cell_nums <- unique(raster_df$cell)
    for (c in cell_nums) {
      
      # Make name for new neuron
      new_name <- paste0("neuron_", c)
      # Grab raster
      c_mask <- raster_df$cell == c
      raster <- raster_df[c_mask, c("time_in_ms", "trial")]
      colnames(raster) <- c("time", "trial")
      raster <- raster[raster$time <= time_cutoff,]
      # Extract meta data
      recording_name <- "not_provided"
      if ("recording_name" %in% colnames(raster_df)) {
        recording_name <- unique(raster_df$recording_name[c_mask])
        if (length(recording_name) != 1) recording_name <- "not_provided"
      } 
      hemi <- "not_provided"
      if ("hemi" %in% colnames(raster_df)) {
        hemi <- unique(raster_df$hemi[c_mask])
        if (length(hemi) != 1) hemi <- "not_provided"
      }
      genotype <- "not_provided"
      if ("genotype" %in% colnames(raster_df)) {
        genotype <- unique(raster_df$genotype[c_mask])
        if (length(genotype) != 1) genotype <- "not_provided"
      }
      sex <- "not_provided"
      if ("sex" %in% colnames(raster_df)) {
        sex <- unique(raster_df$sex[c_mask])
        if (length(sex) != 1) sex <- "not_provided"
      }
      region <- "not_provided"
      if ("region" %in% colnames(raster_df)) {
        region <- unique(raster_df$region[c_mask])
        if (length(region) != 1) region <- "not_provided"
      }
      age <- "not_provided"
      if ("age" %in% colnames(raster_df)) {
        age <- unique(raster_df$age[c_mask])
        if (length(age) != 1) age <- "not_provided"
      }
      
      # Make new neuron object
      new_neuron <- new.neuron(
        id_num = c,
        recording_name = recording_name,
        genotype = genotype,
        sex = sex,
        hemi = hemi,
        region = region,
        age = age,
        unit_time = "ms",
        unit_data = "spike",
        t_per_bin = bin_size,
        sample_rate = sample_rt
      )
      raster <- as.matrix(raster)
      new_neuron$load_spike_raster_R(raster, min_duration, max_displacement)
      
      # Save in list
      neuron_list[[new_name]] <- new_neuron
      
    }
    
    return(neuron_list)
    
  }

# Helper function for plotting
make.plot.title <- function(
    neuron,                   # neuron object
    plot_title
  ) {
    
    # Grab neuron ID info
    id_data <- neuron$fetch_id_data() 
    recording_name <- id_data$recording_name
    id_num <- id_data$id_num
    genotype <- id_data$genotype
    sex <- id_data$sex
    hemi <- id_data$hemi
    region <- id_data$region
    age <- id_data$age
    sim <- id_data$sim
    if (sim) id_num <- paste(id_num, "(sim)")
    
    # Make plot title 
    plot_title <- paste0(
      plot_title,
      ", neuron ", recording_name, ", #", id_num
    )
    # Add covariates
    for (var in c(genotype, sex, hemi, region, age)) {
      if (var != "not provided" && var != "not_provided") {
        plot_title <- paste0(plot_title, ", ", var)
      }
    }
    
    return(plot_title)
    
  }

#' Function to plot autocorrelation of neuron
#' 
#' Wrapper function to fetch and plot autocorrelation information from a neuron object. Assumes the autocorrelation has already been computed. 
#' 
#' @name plot.autocorrelation 
#' @rdname plot-autocorrelation
#' @usage plot.autocorrelation(
#'  nrn, 
#'  plot_title = "Est. autocorr", 
#'  bias_term = 0, 
#'  plot_time_cutoff = Inf, 
#'  return_plot = FALSE
#' )
#' @param nrn Neuron object for which to plot autocorrelation.
#' @param plot_title Title for the plot (default: "Est. autocorr").
#' @param bias_term Bias term to plot as a horizontal line (default: NULL).
#' @param plot_time_cutoff Maximum lag (in bins) to display on the x-axis (default: Inf).
#' @param return_plot Logical indicating whether to return the ggplot object (TRUE) or print it (FALSE) (default: FALSE).
#' @return A ggplot object showing the estimated and fitted autocorrelation.
#' @export
plot.autocorrelation <- function(
    nrn,
    plot_title = "Est. autocorr",
    bias_term = NULL,
    plot_time_cutoff = Inf,
    return_plot = FALSE
  ) {
    
    # Make title
    plot_title <- make.plot.title(nrn, plot_title)
    
    # Fetch estimated and fitted autocorrelation 
    autocorr <- nrn$fetch_autocorr_R()
    autocorr_edf <- nrn$fetch_autocorr_edf_R()
    
    # Make data frame for plotting 
    df_temp <- data.frame(
      autocorrelation = autocorr[2:length(autocorr)],
      bin = seq(from = 1, to = length(autocorr) - 1, by = 1)
    )
    if (length(autocorr_edf) > 1) {
      df_temp$autocorrelation_fitted <- autocorr_edf[2:length(autocorr_edf)]
    }
    df_temp <- df_temp[df_temp$bin <= plot_time_cutoff,]
    
    # Make plot
    title_size <- 14 
    axis_size <- 12 
    legend_size <- 10
    plt <- ggplot2::ggplot(df_temp) +
      ggplot2::geom_line(ggplot2::aes(x = bin, y = autocorrelation), color = "blue") + 
      ggplot2::labs(
        title = plot_title,
        x = "Lag (bins)",
        y = "Autocorrelation"
      ) + 
      ggplot2::theme_minimal() + 
      ggplot2::theme(
        panel.background = ggplot2::element_rect(fill = "white", colour = NA),
        plot.background  = ggplot2::element_rect(fill = "white", colour = NA),
        plot.title = ggplot2::element_text(hjust = 0.5, size = title_size),
        axis.title = ggplot2::element_text(size = axis_size),
        axis.text = ggplot2::element_text(size = axis_size),
        legend.title = ggplot2::element_text(size = legend_size),
        legend.text = ggplot2::element_text(size = legend_size),
        legend.position = "bottom"
      )
    if (!is.null(bias_term)) {
      plt <- plt + ggplot2::geom_hline(yintercept = bias_term, linewidth = 2, linetype = "dotted", color = "darkgray")
    }
    if (length(autocorr_edf) > 1) {
      plt <- plt + ggplot2::geom_line(ggplot2::aes(x = bin, y = autocorrelation_fitted), color = "red")
    }
    if (return_plot) {
      return(plt)
    } else {
      print(plt)
      return(invisible(NULL))
    }
    
  }

#' Function to plot spike raster of neuron
#' 
#' Wrapper function to fetch and plot spike raster from a neuron object. Assumes the spike raster has already been loaded.
#' 
#' @name plot.raster
#' @rdname plot-raster
#' @usage plot.raster(nrn, plot_title = "Spike raster", zero_as_onset = TRUE, return_plot = FALSE)
#' @param nrn Neuron object for which to plot spike raster.
#' @param plot_title Title for the plot (default: "Spike raster").
#' @param zero_as_onset Logical indicating whether to plot a vertical line at time zero to indicate stimulus onset (default: TRUE).
#' @param return_plot Logical indicating whether to return the ggplot object (TRUE) or print it (FALSE) (default: FALSE).
#' @return A ggplot object showing the spike raster.
#' @export
plot.raster <- function(
    nrn,  
    plot_title = "Spike raster",
    zero_as_onset = TRUE,
    return_plot = FALSE
  ) {
    
    # Grab raster and time bounds
    spike.raster <- as.data.frame(nrn$fetch_spike_raster_R())
    total_time <- max(spike.raster$time) - min(spike.raster$time)
    dx <- total_time * 0.0025
    
    # Make plot
    title_size <- 14 
    axis_size <- 12 
    legend_size <- 10
    plt <- ggplot2::ggplot(spike.raster) +
      ggplot2::geom_segment(
        ggplot2::aes(x = time - dx, xend = time + dx, y = trial, yend = trial), 
        color = "black",
        linewidth = 0.5
      ) +
      ggplot2::labs(
        title = make.plot.title(nrn, plot_title),
        x = "Time (ms)",
        y = "Trial"
      ) + 
      ggplot2::theme_minimal()
    if (zero_as_onset && min(spike.raster$time) <= 0) plt <- plt + 
      ggplot2::geom_vline(xintercept = 0, linetype = "solid", linewidth = 1, color = "darkblue") 
    plt <- plt + 
      ggplot2::theme(
        panel.background = ggplot2::element_rect(fill = "white", colour = NA),
        plot.background  = ggplot2::element_rect(fill = "white", colour = NA),
        plot.title = ggplot2::element_text(hjust = 0.5, size = title_size),
        axis.title = ggplot2::element_text(size = axis_size),
        axis.text = ggplot2::element_text(size = axis_size),
        legend.title = ggplot2::element_text(size = legend_size),
        legend.text = ggplot2::element_text(size = legend_size),
        legend.position = "bottom"
      )
    
    if (return_plot) {
      return(plt)
    } else {
      print(plt)
      return(invisible(NULL))
    }
    
  }

#' Helper function for testing assumption of autocorrelation processing
#' 
#' The approach we're using is valid if autocorr (red lines) is below mean firing rate (blue line). If so, then the equation we're using to convert spiking autocorr to guassian sigma should work. 
#' 
#' @param autocor_results Data frame of autocorrelation results from the \code{process.autocorr} function.
#' @return Nothing; this function produces a base-R plot for visual inspection.
#' @export
test.sigma.assumption <- function(
    autocor_results
  ) {
    
    autocor_results <- autocor_results[order(autocor_results$lambda_bin),]
    these_cols <- c(
      "lambda_bin",
      "max_autocorr",
      "mean_autocorr",
      "min_autocorr"
    )
    plot(
      autocor_results[,2], 
      ylim = c(0, max(autocor_results[,these_cols])*1.1),
      type = "l",
      ylab = "value",
      xlab = "neuron (sorted by lambda_bin)",
      col = "blue"
    )
    lines(autocor_results$max_autocorr, col = "red4")
    lines(autocor_results$mean_autocorr, col = "red3")
    lines(autocor_results$min_autocorr, col = "red")
    
  }

#' Function to process autocorrelation of neuron list
#' 
#' First estimates autocorrelation of each neuron in the list, then fits an exponential decay function to the estimated autocorrelation, and finally collects and returns summary statistics for each neuron.
#' 
#' @param neuron_list An R list of neuron objects.
#' @param bin_count_action Method for counting spikes in each bin when computing autocorrelation; one of "boolean", "mean", or "sum" (default: "sum").
#' @param max_lag Maximum lag (in units of the trial data) to compute autocorrelation; if 0, uses the number of time bins in the neuron's trial data (default: 0).
#' @param A0 Initial guess for amplitude parameter of exponential decay function (default: 0.001).
#' @param tau0 Initial guess for time constant parameter of exponential decay function (default: 1.0).
#' @param ctol Convergence tolerance for fitting exponential decay function (default: 1e-8).
#' @param max_evals Maximum number of evaluations for fitting exponential decay function (default: 500).
#' @param check_autofiring_ratio Logical indicating whether to check the assumption that autocorrelation values are below the mean firing rate with \code{test.sigma.assumption} function (default: FALSE).
#' @param print_plots Logical indicating whether to print autocorrelation plots for each neuron (default: FALSE).
#' @param plot_time_cutoff Maximum lag (in bins) to display on the x-axis of plots (default: Inf).
#' @param use_raw Logical indicating whether to use raw autocorrelation (true) or standard centered and normalized correlation (false) (default: TRUE).
#' @return A data frame with one row per neuron and columns for lambda (mean spike rate per ms and per bin), amplitude (A), time constant (tau), bias term, first autocorrelation value, maximum autocorrelation value, mean autocorrelation value, and minimum autocorrelation value.
#' @export
process.autocorr <- function(
    neuron_list,
    bin_count_action = "sum", 
    max_lag = 0,
    A0 = 0.001,
    tau0 = 1.0,
    ctol = 1e-8,
    max_evals = 500,
    check_autofiring_ratio = FALSE,
    print_plots = FALSE,
    plot_time_cutoff = Inf,
    use_raw = TRUE
  ) {
    
    # Array to hold results
    autocor_results <- c()
    
    # Find max lag if needed
    if (max_lag == 0) {
      trial_lengths <- rep(Inf, length(neuron_list))
      for (i in seq_along(neuron_list)) {
        nrn <- neuron_list[[i]]
        trial_lengths[i] <- nrow(nrn$fetch_trial_data_R())
      }
      max_lag <- min(trial_lengths)
    }
    
    # Loop through neurons in the list
    for (i in seq_along(neuron_list)) {
      
      # Grab neuron
      nrn <- neuron_list[[i]]
      
      # Set exponential decay function (EDF) parameters 
      nrn$set_edf_initials(A0, tau0)
      nrn$set_edf_termination(ctol, max_evals)
      
      # Compute autocorrelation
      nrn$compute_autocorrelation(bin_count_action, max_lag, use_raw)
      
      # Fit autocorrelation 
      nrn$fit_autocorrelation()
      
      # Fetch results 
      edf_results <- nrn$fetch_EDF_parameters()
      # ... get amplitude
      A <- edf_results["A"]
      # ... get time constant
      tau <- edf_results["tau"]
      # ... get bias term
      bias_term <- edf_results["bias_term"]
      # ... get lambda (mean spike rate, per ms and per bin)
      lambda <- nrn$fetch_lambda()
      # ... get series of EDF-fitted autocorrelation values
      autocorr_edf <- nrn$fetch_autocorr_edf_R()
      
      # Collect results and separate out the first predicted autocorrelation value
      autocor_results_r <- c(
        lambda, # vector of length two
        A,
        tau,
        bias_term,
        autocorr_edf[1],
        max(autocorr_edf[2:length(autocorr_edf)]),
        mean(autocorr_edf[2:length(autocorr_edf)]), 
        min(autocorr_edf[2:length(autocorr_edf)])
      )
      
      # Save results with those from other neurons
      autocor_results <- rbind(autocor_results, autocor_results_r)
      
      # Print plots 
      if (print_plots) {
        
        # Grab neuron ID info
        id_data <- nrn$fetch_id_data() 
        recording_name <- id_data$recording_name
        sim <- id_data$sim
        if (sim) recording_name <- paste(recording_name, "(sim)")
        
        print(plot.autocorrelation(nrn, "Est. autocorr", bias_term, plot_time_cutoff))
        
      }
      
    }
    
    autocor_results <- data.frame(names(neuron_list), autocor_results)
    
    rownames(autocor_results) <- NULL
    colnames(autocor_results) <- c(
      "cell",
      "lambda_ms",
      "lambda_bin",
      "A", 
      "tau", 
      "bias_term",
      "autocorr1",
      "max_autocorr",
      "mean_autocorr",
      "min_autocorr"
    )
    
    # Convert to data frame 
    if (check_autofiring_ratio) {
      test.sigma.assumption(autocor_results)
    }
    
    return(autocor_results)
    
  }

#' Function to estimate autocorrelation parameters using dichotomized Gaussian simulations
#' 
#' This function performs the same estimate-and-fit procedure as \code{process.autocorr}, but does so multiple times for each neuron using simulated spike rasters generated from a dichotomized Gaussian model of that neuron. 
#' 
#' @param neuron_list An R list of neuron objects.
#' @param n_trials_per_sim Number of trials to simulate for each simulation (default: 300).
#' @param n_sims_per_neurons Number of simulations to run for each neuron (default: 100).
#' @param bin_count_action Method for counting spikes in each bin when computing autocorrelation; one of "boolean", "mean", or "sum" (default: "sum").
#' @param A0 Initial guess for amplitude parameter of exponential decay function (default: 0.001).
#' @param tau0 Initial guess for time constant parameter of exponential decay function (default: 1.0).
#' @param ctol Convergence tolerance for fitting exponential decay function (default: 1e-8).
#' @param max_evals Maximum number of evaluations for fitting exponential decay function (default: 500).
#' @param use_raw Logical indicating whether to use raw autocorrelation (true) or standard centered and normalized correlation (false) (default: TRUE).
#' @return A list containing a data frame of autocorrelation parameter estimates (one row per simulation), a data frame of neuron identifiers (one row per neuron), and the number of simulations run per neuron.
#' @export
estimate.autocorr.params <- function(
    neuron_list,
    n_trials_per_sim = 300, 
    n_sims_per_neurons = 100, 
    bin_count_action = "sum", # "boolean", "mean", or "sum" 
    max_lag = 0,
    A0 = 0.001,
    tau0 = 1.0,
    ctol = 1e-8,
    max_evals = 500,
    use_raw = TRUE
  ) {
    
    # Find max lag if needed
    if (max_lag == 0) {
      trial_lengths <- rep(Inf, length(neuron_list))
      for (i in seq_along(neuron_list)) {
        nrn <- neuron_list[[i]]
        trial_lengths[i] <- nrow(nrn$fetch_trial_data_R())/nrn$fetch_id_data()[["t_per_bin"]]
      }
      max_lag <- min(trial_lengths)
    }
   
    # Pre-allocate matrix 
    n_neurons <- length(neuron_list)
    dg_estimates <- matrix(NA_real_, nrow = n_sims_per_neurons * n_neurons, ncol = 9)
    
    # Initialize neuron id df
    neuron_id <- data.frame()
    
    # Loop through neurons in the list
    for (i in seq_along(neuron_list)) {
      
      # Get neuron 
      nrn <- neuron_list[[i]]
      
      # Get neuron id 
      nrn_id <- as.data.frame(lapply(nrn$fetch_id_data(), rep, times = 1))
      neuron_id <- rbind(neuron_id, nrn_id)
      
      # Run simulations to estimate value for this neuron and load into matrix
      row_idx_initial <- (i - 1) * n_sims_per_neurons + 1
      row_idx_final <- i * n_sims_per_neurons
      row_idx <- row_idx_initial:row_idx_final
      dg_estimates[row_idx,] <- nrn$estimate_autocorr_params(
        n_trials_per_sim,
        n_sims_per_neurons,
        max_lag,
        bin_count_action,
        A0,
        tau0,
        ctol,
        max_evals,
        use_raw,
        FALSE
      )
      
    }
    
    dg_estimates <- as.data.frame(dg_estimates)
    colnames(dg_estimates) <- c(
      "lambda_ms",
      "lambda_bin",
      "A", 
      "tau", 
      "bias_term",
      "autocorr1",
      "max_autocorr",
      "mean_autocorr",
      "min_autocorr"
    )
    
    return(list(
      estimates = dg_estimates,
      neuron_id = neuron_id,
      n_sims_per_neurons = n_sims_per_neurons
    ))
    
  }

#' Function to analyze autocorrelation estimates with bootstrapping
#' 
#' This function takes the output from \code{estimate.autocorr.params} and performs a bootstrap analysis to estimate the distribution of the mean time constant (tau) for different levels of specified covariates.
#' 
#' @param ests Output from \code{estimate.autocorr.params}, or a list of such outputs.
#' @param covariate Character string or vector of character strings specifying the covariate(s) to analyze (e.g., "hemi").
#' @param n_bs Number of bootstrap resamples to perform (default: 1e4).
#' @return A list with two elements: 
#'  \describe{
#'    \item{resamples}{data frame with bootstrap resamples of the mean tau for each combination of covariate levels}
#'    \item{distribution_plot}{ggplot object showing the distribution of mean tau for each combination of covariate levels}
#'  }
analyze.autocorr <- function(
    ests,
    covariate,
    n_bs = 1e4
  ) {
    
    # Check input types
    error_message <- "Input is not a valid estimate.autocorr.params output or list of such output."
    if (class(ests) == "list") {
      if ("n_sims_per_neurons" %in% names(ests)) {
        if (!(all(covariate %in% names(ests$neuron_id)))) {
          stop("Covariate not found in neuron_id.")
        }
        # ... this is single input, wrap in list
        ests <- list(ests)
      } else {
        # ... check if list of proper inputs 
        for (i in seq_along(ests)) {
          if (!("n_sims_per_neurons" %in% names(ests[[i]]))) {
            stop(error_message)
          }
          if (!(all(covariate %in% names(ests[[i]]$neuron_id)))) {
            stop("Covariate not found in neuron_id.")
          }
        }
      }
    } else {
      stop(error_message)
    }
    
    # Combine batches into big data frame
    ests_all <- data.frame()
    for (i in seq_along(ests)) {
      
      # Grab estimate list 
      est <- ests[[i]]
      
      # Expand neuron_id 
      est$neuron_id <- est$neuron_id[rep(1:nrow(est$neuron_id), each = est$n_sims_per_neurons), ]
      
      # Combine with estimates 
      estimates_id <- cbind(est$estimates, est$neuron_id)
      
      # Add to big data frame
      ests_all <- rbind(ests_all, estimates_id)
      
    }
    
    # Get covariate levels and their interactions
    est_covariates <- list() 
    est_covariate_levels <- list()
    for (c in c(covariate)) {
      est_covariates[[c]] <- ests_all[,c]
      est_covariate_levels[[c]] <- unique(ests_all[,c])
    }
    cov_X <- do.call(expand.grid, list(est_covariate_levels, stringsAsFactors = FALSE))
    n_interX <- nrow(cov_X)
    
    # Run bootstraps 
    resamples <- matrix(NA, nrow = n_bs, ncol = n_interX)
    n_est <- nrow(ests_all)
    interX_names <- c()
    for (i in 1:n_interX) {
      
      lvl <- cov_X[i,]
      interX_names <- c(interX_names, paste0(lvl, collapse = "_"))
      lvl_mask <- rep(TRUE, n_est)
      
      if (length(lvl) == 1) {
        lvl_mask <- lvl_mask & ests_all[,c] == lvl
      } else {
        for (c in c(covariate)) {
          lvl_mask <- lvl_mask & ests_all[,c] == cov_X[i,c]
        }
      }
      
      if (any(lvl_mask)) {
        n_cells <- length(unique(ests_all$id_num))
        for (j in 1:n_bs) {
          # Net effect is to resample from each neuron's estimates in a way that 
          #   accounts for the uncertainty estimated by the DG. E.g., if there are n neurons, 
          #   each resample has a 1/n chance of drawing from a given neuron N; but, as N
          #   is represented by n_sims values from the DG simulations, the probability of a given value
          #   being drawn to represent N is determined by the simulations. 
          these_samples <- sample(ests_all[lvl_mask, "tau"], n_cells, replace = TRUE)
          resamples[j,i] <- mean(these_samples)
        }
      }
      
    }
    resamples <- as.data.frame(resamples)
    colnames(resamples) <- interX_names
    
    resamples <- resamples[, colSums(!is.na(resamples)) > 0]
    
    # Plot
    df <- data.frame(
      value = unlist(resamples, use.names = FALSE),
      group = rep(names(resamples), each = nrow(resamples))
    )
    
    # Plot
    title_size <- 14 
    axis_size <- 12 
    legend_size <- 10
    distribution_plot <- ggplot2::ggplot(df, ggplot2::aes(x = value, fill = group)) +
      ggplot2::geom_density(alpha = 0.6) +
      #scale_y_continuous(transform = "log1p") +
      ggplot2::labs(title = "Expected Time Constant", x = "ms", y = "Density") +
      ggplot2::theme_minimal() + 
      ggplot2::theme(
        panel.background = ggplot2::element_rect(fill = "white", colour = NA),
        plot.background  = ggplot2::element_rect(fill = "white", colour = NA),
        plot.title = ggplot2::element_text(hjust = 0.5, size = title_size),
        axis.title = ggplot2::element_text(size = axis_size),
        axis.text = ggplot2::element_text(size = axis_size),
        legend.title = ggplot2::element_text(size = legend_size),
        legend.text = ggplot2::element_text(size = legend_size),
        legend.position = "bottom"
      ) 
    
    return(
      list(
        resamples = resamples,
        distribution_plot = distribution_plot
      )
    )
    
  }

#' Compute autocorrelation of single neuron object
#' 
#' This function computes the autocorrelation of a single neuron object using specified parameters. It's a wrapper around the \code{compute_autocorrelation} method of the neuron class.
#' 
#' @param nrn Neuron object for which to compute autocorrelation.
#' @param bin_count_action Method for counting spikes in each bin when computing autocorrelation; one of "boolean", "mean", or "sum" (default: "sum").
#' @param max_lag Maximum lag (in units of the trial data) to compute autocorrelation; if 0, uses the number of time bins in the neuron's trial data (default: 0).
#' @param use_raw Logical indicating whether to use raw autocorrelation (true) or standard centered and normalized correlation (false) (default: TRUE).
#' @return None. Modifies neuron in place.
#' @export
compute.autocorr <- function(
    nrn,
    bin_count_action = "sum",
    max_lag = 0,
    use_raw = TRUE
  ) {
    if (max_lag == 0) {
      max_lag <- nrow(nrn$fetch_trial_data_R())
    }
    nrn <- nrn$compute_autocorrelation(bin_count_action, max_lag, use_raw)
  }

#' Fit exponential decay function to autocorrelation of single neuron object
#' 
#' This function fits an exponential decay function to the autocorrelation of a single neuron object using specified parameters. It's a wrapper around the \code{fit_autocorrelation} method of the neuron class.
#'
#' @param nrn Neuron object for which to fit autocorrelation.
#' @param A0 Initial guess for amplitude parameter of exponential decay function (default: 0.001).
#' @param tau0 Initial guess for time constant parameter of exponential decay function (default: 1.0).
#' @param ctol Convergence tolerance for fitting exponential decay function (default: 1e-8).
#' @param max_evals Maximum number of evaluations for fitting exponential decay function (default: 500).
#' @return None. Modifies neuron in place.
#' @export
fit.edf.autocorr <- function(
    nrn,
    A0 = 0.001,
    tau0 = 1.0,
    ctol = 1e-8,
    max_evals = 500
  ) {
    nrn$set_edf_initials(A0, tau0)
    nrn$set_edf_termination(ctol, max_evals)
    nrn <- nrn$fit_autocorrelation()
  }

