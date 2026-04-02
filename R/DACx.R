
# Intro ################################################################################################################

# Neurons: A framework for neuron modelling, simulation, and analysis

# By Mike Barkasi
# GNU GPLv3: https://www.gnu.org/licenses/gpl-3.0.en.html
#   Copyright (c) 2025

#' @useDynLib neurons, .registration = TRUE
#' @import Rcpp
#' @import RcppEigen 
#' @import ggplot2
NULL

.onLoad <- function(libname, pkgname) {
    Rcpp::loadModule("motif", TRUE)
    Rcpp::loadModule("network", TRUE)
    Rcpp::loadModule("Projection", TRUE)
    init_known_celltypes()
  }

# Initialization for C++ object classes ################################################################################

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
