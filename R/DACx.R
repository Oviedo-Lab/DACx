
# Intro ################################################################################################################

# Neurons: A framework for neuron modelling, simulation, and analysis

# By Mike Barkasi
# GNU GPLv3: https://www.gnu.org/licenses/gpl-3.0.en.html
#   Copyright (c) 2025

#' @useDynLib DACx, .registration = TRUE
#' @import Rcpp
#' @import RcppEigen 
#' @import ggplot2
#' @import plotly
NULL

.onLoad <- function(libname, pkgname) {
    Rcpp::loadModule("motif",      TRUE)
    Rcpp::loadModule("network",    TRUE)
    Rcpp::loadModule("Projection", TRUE)
  }

# Initialization for C++ object classes ################################################################################

#' Initialize network (circuit) motif
#' 
#' This function initializes a new motif object with specified parameters. Motifs are used for building networks of interconnected neurons. They are recipes for building internode projections within a neural network. They are "columnar", in the sense that they are repeated across cortical columns. 
#' 
#' @param motif_name Character string giving name of the motif (default: "not_provided").
#' @param hemi Hemisphere to which the motif applies: use 0 or "left" for left, 1 or "right" for right, and -1, "all", or "both" for left and right.
#' @return A new motif object.
#' @export
new.motif <- function(
    motif_name = "not_provided",
    hemi       = "both"
  ) {
    if (hemi != "both" && hemi != "all" && hemi != "left" && hemi != "right" && hemi != -1 && hemi != 0 && hemi != 1) {
      stop("Value of hemi must be: \n\t0 or 'left' for left \n\t1 or 'right' for right \n\t -1, 'all', or 'both' for both")
    }
    if (hemi == "both" || hemi == "all") hemi <- -1
    if (hemi == "left" ) hemi <- 0
    if (hemi == "right") hemi <- 1
    motif <- new(
      motif,
      motif_name,
      hemi
    )
    return(motif)
  }

#' Initialize neuron network
#' 
#' This function initializes a new network object with specified parameters. Networks are used to simulate two-dimensional cortical patches (of layers and columns) using Growth Transform dynamical systems. 
#' 
#' Mathematically, networks are points (representing neurons) connected by directed edges. Within the growth-transform (GT) model framework, these edges are synaptic conductance values representing synaptic connections between neurons.
#' 
#' Point types: Points can be grouped by types, which affect their behavior and connectivity. Within the GT model framework, these types each have their own temporal modulation constants (determining, e.g., whether the cell bursts or fires singular spikes) and valence (excitatory or inhibitory).
#' 
#' Global structure: Modelling the mammalian cortex, networks are assumed to divide into a coarse-grained two-dimensional coordinate system of layers (rows) and columns (columns). Each point is assigned to a layer-column coordinate (called a "node"), having both local x-y coordinates within that node and a global x-y coordinate within the network. 
#'  
#' Local structure: Each layer-column coordinate defines a "node" containing a number of points determined by layer and type. Connections (edges) within a node are determined by a local recurrence factor matrix determining the synaptic conductance between points of each type. These edges are called "local". 
#' 
#' Long-range projections: Connections (edges) between points in different nodes are determined by a long-range projection motif and labelled with the same of that motif. 
#' 
#' @return A new network object.
#' @export
new.network <- function() {new(network)}

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
#' @param type_name Character string giving name of the cell type, e.g. "pyramidal", "PV", "SST", etc.
#' @return List of parameters for the named cell type. 
#' @export
fetch.cell.type.params <- function(type_name) fetch_cell_type_params(type_name)

#' Add new cell type
#' 
#' This function adds a user-defined cell type to the current session. It's just a wrapper for the Rcpp-exported \code{add_cell_type} function. Technically, \code{cell_type} is a struct defined in the Rcpp backend of the DACx package. They are essentially labeled lists whose fields are described by the parameters below. Each session stores cell types in the Rcpp backend in an \code{unordered_map} with string labels. All parameters come with biologically realistic (and mathematically workable) default values, except for \code{type_name}, \code{synaptic_conductance}, and \code{equilibrium_potential}. 
#' 
#' @param type_name Character string giving name of the cell type, e.g. "pyramidal", "PV", "SST", etc.
#' @param synaptic_conductance Conductance (nS) of this cell type's synapses, treating this cell type as post-synaptic and indexing by the type of each possible pre-synaptic cell. Can be a single number (applied uniformly to every pre-synaptic type), a numeric vector ordered as in \code{print.known.celltypes()}, or a named list keyed by pre-synaptic type name, e.g. \code{list(pyramidal = 0.15, PV = 0.08)}.
#' @param equilibrium_potential Induced potential (mV) at which no current naturally flows across this cell type's membrane, given the neurotransmitter released by each possible pre-synaptic cell type (e.g., 0 mV for an excitatory pre-synaptic cell, -70 mV for an inhibitory pre-synaptic cell). Accepts the same formats as \code{synaptic_conductance}.
#' @param tau_syn Decay time constant (ms) of the post-synaptic current evoked by the neurotransmitter of each possible pre-synaptic cell type (e.g., faster for AMPA, slower for GABA_A). A larger value makes the current outlast the pre-synaptic spike; 0 recovers an instantaneous (boxcar) current. Accepts the same formats as \code{synaptic_conductance}. Default is 2.0.
#' @param tau_fast Time constant (ms) of the fast sodium (Na+) current (Na+ influx is inward, i.e. negative under the outward-positive convention; time to flow in). Default is 1.0.
#' @param tau_slow Time constant (ms) of the slow calcium (Ca2+) current (Ca2+ influx is inward, i.e. negative under the outward-positive convention; time to pump out). Default is 60.0.
#' @param tau_Vs Time constant (ms) for restoring presynaptic vesicles, i.e., recovery from short-term depression (STD). Default is 100.0.
#' @param I_slow Slow-current molecule (e.g., Ca2+) influx as concentration per spike (concentration/spike). Default is 0.01.
#' @param U_Vs Utilization ratio (concentration/spike) of vesicles per spike. Default is 0.05.
#' @param max_spike_rate Constant (spikes/ms) controlling estimation of spike rate and its maximum value. Default is 0.1.
#' @param transmission_velocity Transmission velocity (in microns/ms) along axon, for each neuron type. Default value is 30e3.
#' @param spine_density Scale controlling percentage of dendrite nodes with spines: zero means none, one means all. Default is 0.0. 
#' @param axon_target Character string giving target of axon projections, one of: "spine", "dendrite_shaft", "soma", or "axon_shaft". Default is "dendrite_shaft".
#' @param I_spike Spike current, in pA. Default value is 1e3 (i.e., 1 nA).
#' @param dHdv_bound Bound on derivative of metabolic energy wrt potential, such that \code{dHdv_bound > abs(dHdv)}, in pA, for each neuron in the network, based on its type. Default value is 1.05e3.
#' @param spike_potential Peak potential during a spike, in mV. Default value is 35.0.
#' @param spike_width Time spike activates synapse, in ms. Default value is 1.0. 
#' @param resting_potential Resting potential, in mV; absolute value plus a little bit used as \code{v_bound}. Default value is -70.0.
#' @param v_bound Multiplier on the absolute value of \code{resting_potential} giving the membrane potential barrier (mirrors \code{dHdv_bound}). Increase to allow hyperpolarization below rest. Default is 1.15.
#' @param threshold Spike threshold, in mV. Default value is -55.0.
#' @param leak_conductance Conductance controlling the leak current, \code{I_leak = leak_conductance * (v - resting_potential)} (outward-positive), in nS. Default value is 10.0.
#' @param axon_branch_count Expected number of axon branches. Default is 10. 
#' @param dendrite_branch_count Expected number of dendrite branches. Default is 10. 
#' @param branch_independence Scale between 0 and 1; 0 = all branches connect to soma from single segment, 1 = all branches connect directly to soma. Default is 0.5.
#' @param branch_spread Scale between 0 and 1; 0 = no tendency to extend away from soma, 1 = straight line away from soma. Default is 0.5.
#' @param apical_target_layer Character string giving target layer for apical dendrites. Default is "none".
#' @return Nothing.
#' @export
add.cell.type <- function(
    type_name,
    synaptic_conductance  = 0.0,
    equilibrium_potential = 0.0,
    tau_syn               = 2.0,
    tau_fast              = 1.0,
    tau_slow              = 60.0,
    tau_Vs                = 100.0,
    I_slow                = 0.01, 
    U_Vs                  = 0.05, 
    max_spike_rate        = 0.1,
    transmission_velocity = 30e3,
    spine_density         = 0.0,
    axon_target           = "dendrite_shaft",
    I_spike               = 1e3,
    dHdv_bound            = 1.05,
    spike_potential       = 35.0,
    spike_width           = 1.0, 
    resting_potential     = -70.0,
    v_bound               = 1.15,
    threshold             = -55.0,
    leak_conductance      = 10.0,
    axon_branch_count     = 10L,
    dendrite_branch_count = 10L,
    branch_independence   = 0.5,
    branch_spread         = 0.5,
    apical_target_layer   = "none"
  ) {
    add_cell_type(list(
      type_name              = type_name,
      synaptic_conductance   = synaptic_conductance,
      equilibrium_potential  = equilibrium_potential,
      tau_syn                = tau_syn,
      tau_fast               = tau_fast, 
      tau_slow               = tau_slow, 
      tau_Vs                 = tau_Vs, 
      I_slow                 = I_slow, 
      U_Vs                   = U_Vs, 
      max_spike_rate         = max_spike_rate,
      transmission_velocity  = transmission_velocity,
      spine_density          = spine_density,
      axon_target            = axon_target,
      I_spike                = I_spike,
      dHdv_bound             = dHdv_bound,
      spike_potential        = spike_potential,
      spike_width            = spike_width, 
      resting_potential      = resting_potential,
      v_bound                = v_bound,
      threshold              = threshold,
      leak_conductance       = leak_conductance,
      axon_branch_count      = as.integer(axon_branch_count),
      dendrite_branch_count  = as.integer(dendrite_branch_count),
      branch_independence    = branch_independence,
      branch_spread          = branch_spread,
      apical_target_layer    = apical_target_layer
    ))
  }

#' Modify existing cell type 
#' 
#' This function modifies parameters of an existing cell type in the current session. Parameters can be updated selectively. If a parameter is not specified (or is specified as \code{NULL}), the existing value will be kept.
#' 
#' @param old_type_name Character string giving name of the cell type, e.g. "pyramidal", "PV", "SST", etc.
#' @param synaptic_conductance Conductance (nS) of this cell type's synapses, treating this cell type as post-synaptic and indexing by the type of each possible pre-synaptic cell. Can be a single number (applied uniformly to every pre-synaptic type), a numeric vector ordered as in \code{print.known.celltypes()}, or a named list keyed by pre-synaptic type name, e.g. \code{list(pyramidal = 0.15, PV = 0.08)}.
#' @param equilibrium_potential Induced potential (mV) at which no current naturally flows across this cell type's membrane, given the neurotransmitter released by each possible pre-synaptic cell type (e.g., 0 mV for an excitatory pre-synaptic cell, -70 mV for an inhibitory pre-synaptic cell). Accepts the same formats as \code{synaptic_conductance}.
#' @param tau_syn Decay time constant (ms) of the post-synaptic current evoked by the neurotransmitter of each possible pre-synaptic cell type (e.g., faster for AMPA, slower for GABA_A). A larger value makes the current outlast the pre-synaptic spike; 0 recovers an instantaneous (boxcar) current. Accepts the same formats as \code{synaptic_conductance}.
#' @param tau_fast Time constant (ms) of the fast sodium (Na+) current (Na+ influx is inward, i.e. negative under the outward-positive convention; time to flow in).
#' @param tau_slow Time constant (ms) of the slow calcium (Ca2+) current (Ca2+ influx is inward, i.e. negative under the outward-positive convention; time to pump out).
#' @param tau_Vs Time constant (ms) for restoring presynaptic vesicles, i.e., recovery from short-term depression (STD).
#' @param I_slow Slow-current molecule (e.g., Ca2+) influx as concentration per spike (concentration/spike).
#' @param U_Vs Utilization ratio (concentration/spike) of vesicles per spike.
#' @param max_spike_rate Constant (spikes/ms) controlling estimation of spike rate and its maximum value.
#' @param transmission_velocity Transmission velocity (in microns/ms) along axon, for each neuron type.
#' @param spine_density Scale controlling percentage of dendrite nodes with spines: zero means none, one means all.
#' @param axon_target Character string giving target of axon projections, one of: "spine", "dendrite_shaft", "soma", or "axon_shaft".
#' @param I_spike Spike current, in pA.
#' @param dHdv_bound Scale factor giving the bound on derivative of metabolic energy wrt potential, such that \code{dHdv_bound * I_spike > abs(dHdv)}, for each neuron in the network, based on its type.
#' @param spike_potential Peak potential during a spike, in mV.
#' @param spike_width Time spike activates synapse, in ms. 
#' @param resting_potential Resting potential, in mV; absolute value plus a little bit used as \code{v_bound}.
#' @param v_bound Multiplier on the absolute value of \code{resting_potential} giving the membrane potential barrier (mirrors \code{dHdv_bound}). Increase to allow hyperpolarization below rest.
#' @param threshold Spike threshold, in mV.
#' @param leak_conductance Conductance controlling the leak current, \code{I_leak = leak_conductance * (v - resting_potential)} (outward-positive), in nS.
#' @param axon_branch_count Expected number of axon branches.
#' @param dendrite_branch_count Expected number of dendrite branches.
#' @param branch_independence Scale between 0 and 1; 0 = all branches connect to soma from single segment, 1 = all branches connect directly to soma.
#' @param branch_spread Scale between 0 and 1; 0 = no tendency to extend away from soma, 1 = straight line away from soma.
#' @param apical_target_layer Character string giving target layer for apical dendrites.
#' @return Nothing.
#' @export
modify.cell.type <- function(
    old_type_name,
    new_type_name           = NULL,
    synaptic_conductance    = NULL,
    equilibrium_potential   = NULL,
    tau_syn                 = NULL,
    tau_fast                = NULL,
    tau_slow                = NULL,
    tau_Vs                  = NULL,
    I_slow                  = NULL, 
    U_Vs                    = NULL, 
    max_spike_rate          = NULL,
    transmission_velocity   = NULL,
    spine_density           = NULL,
    axon_target             = NULL,
    I_spike                 = NULL,
    dHdv_bound              = NULL,
    spike_potential         = NULL,
    spike_width             = NULL,
    resting_potential       = NULL,
    v_bound                 = NULL,
    threshold               = NULL,
    leak_conductance        = NULL,
    axon_branch_count       = NULL,
    dendrite_branch_count   = NULL,
    branch_independence     = NULL,
    branch_spread           = NULL,
    apical_target_layer     = NULL
  ) {
    ep <- fetch.cell.type.params(old_type_name)  # existing params
    if (is.null(tau_syn))                tau_syn               <- ep$tau_syn
    if (is.null(tau_fast))               tau_fast              <- ep$tau_fast
    if (is.null(tau_slow))               tau_slow              <- ep$tau_slow
    if (is.null(tau_Vs))                 tau_Vs                <- ep$tau_Vs
    if (is.null(I_slow))                 I_slow                <- ep$I_slow
    if (is.null(U_Vs))                   U_Vs                  <- ep$U_Vs
    if (is.null(max_spike_rate))         max_spike_rate        <- ep$max_spike_rate
    if (is.null(transmission_velocity))  transmission_velocity <- ep$transmission_velocity
    if (is.null(spine_density))          spine_density         <- ep$spine_density
    if (is.null(axon_target))            axon_target           <- ep$axon_target
    if (is.null(I_spike))                I_spike               <- ep$I_spike
    if (is.null(dHdv_bound))             dHdv_bound            <- ep$dHdv_bound
    if (is.null(spike_potential))        spike_potential       <- ep$spike_potential
    if (is.null(spike_width))            spike_width           <- ep$spike_width
    if (is.null(resting_potential))      resting_potential     <- ep$resting_potential
    if (is.null(v_bound))                v_bound               <- ep$v_bound
    if (is.null(threshold))              threshold             <- ep$threshold
    if (is.null(leak_conductance))       leak_conductance      <- ep$leak_conductance
    if (is.null(axon_branch_count))      axon_branch_count     <- ep$axon_branch_count
    if (is.null(dendrite_branch_count))  dendrite_branch_count <- ep$dendrite_branch_count
    if (is.null(branch_independence))    branch_independence   <- ep$branch_independence
    if (is.null(branch_spread))          branch_spread         <- ep$branch_spread
    if (is.null(apical_target_layer))    apical_target_layer   <- ep$apical_target_layer
    if (is.null(synaptic_conductance))   synaptic_conductance  <- ep$synaptic_conductance
    if (is.null(equilibrium_potential))  equilibrium_potential <- ep$equilibrium_potential
    if (is.null(new_type_name)) {
      modify_cell_type(old_type_name, list(
        type_name              = old_type_name,
        synaptic_conductance   = synaptic_conductance,
        equilibrium_potential  = equilibrium_potential,
        tau_syn                = tau_syn,
        tau_fast               = tau_fast, 
        tau_slow               = tau_slow, 
        tau_Vs                 = tau_Vs, 
        I_slow                 = I_slow, 
        U_Vs                   = U_Vs, 
        max_spike_rate         = max_spike_rate,
        transmission_velocity  = transmission_velocity,
        spine_density          = spine_density,
        axon_target            = axon_target,
        I_spike                = I_spike,
        dHdv_bound             = dHdv_bound,
        spike_potential        = spike_potential,
        spike_width            = spike_width, 
        resting_potential      = resting_potential,
        v_bound                = v_bound,
        threshold              = threshold,
        leak_conductance       = leak_conductance,
        axon_branch_count      = as.integer(axon_branch_count),
        dendrite_branch_count  = as.integer(dendrite_branch_count),
        branch_independence    = branch_independence,
        branch_spread          = branch_spread,
        apical_target_layer    = apical_target_layer
      ))
    } else {
      add_cell_type(list(
        type_name              = new_type_name,
        synaptic_conductance   = synaptic_conductance,
        equilibrium_potential  = equilibrium_potential,
        tau_syn                = tau_syn,
        tau_fast               = tau_fast, 
        tau_slow               = tau_slow, 
        tau_Vs                 = tau_Vs, 
        I_slow                 = I_slow, 
        U_Vs                   = U_Vs, 
        max_spike_rate         = max_spike_rate,
        transmission_velocity  = transmission_velocity,
        spine_density          = spine_density,
        axon_target            = axon_target,
        I_spike                = I_spike,
        dHdv_bound             = dHdv_bound,
        spike_potential        = spike_potential,
        spike_width            = spike_width, 
        resting_potential      = resting_potential,
        v_bound                = v_bound,
        threshold              = threshold,
        leak_conductance       = leak_conductance,
        axon_branch_count      = as.integer(axon_branch_count),
        dendrite_branch_count  = as.integer(dendrite_branch_count),
        branch_independence    = branch_independence,
        branch_spread          = branch_spread,
        apical_target_layer    = apical_target_layer
      ))
    }
    
  }

# Functions for network ################################################################################################

#' Return principal neurons by standard layers 
#' 
#' This function returns the "principal" (i.e., most common) types of neuron, from among the default known list, for each of the default known layers. 
#' 
#' @param print_nicely Logical; if TRUE, returns nothing while printing the principal neuron types in a nice format. If FALSE, returns a list of principal neuron types by layer, with layer names as keys and neuron type names as values (default: FALSE).
#' @return A list of principal neuron types by layer, with layer names as keys and neuron type names as values.
#' @export
principal.neurons <- function(print_nicely = FALSE) {
    p_list <- list(
      thalamus = "thalmacortical",
      layer    = "spiny_stellate",
      L1       = "neurogliaform_cell",
      L2       = "pyramidal",
      L3       = "pyramidal",
      L4       = "spiny_stellate",
      L5       = "pyramidal",
      L6       = "pyramidal_L6"
    )
    if (print_nicely) {
      cat("Principal neuron types by layer:\n")
      for (layer in names(p_list)) {
        cat("\t", paste0(layer, ": ", p_list[[layer]], "\n"))
      }
      return(invisible(NULL))
    } else {
      return(p_list)
    }
  }

#' Load projection into motif
#' 
#' This function loads a projection schema into a motif object. Projections define internode connectivity within a network built using the motif.
#' 
#' @param motif Motif object into which to load the projection.
#' @param presynaptic_layer Character string giving layer of presynaptic neuron, e.g. "L1", "L2", "L3", "L4", etc.
#' @param postsynaptic_layer Character string, or vector of character strings, giving layer of postsynaptic neuron.
#' @param pre_neuron_fraction Numeric between 0 and 1 giving the fraction of eligible presynaptic neurons that send axons in this projection (default: 0.5). This controls projection sparsity; conductance values are automatically looked up from neuron type properties.
#' @param presynaptic_type Character string giving type of presynaptic neuron (default: "principal").
#' @param postsynaptic_type Character string giving type of postsynaptic neuron (default: "principal").
#' @param max_col_shift_up Maximum number of columns upwards (increasing columnar indexes) that the projection can reach (default: 0, should be positive integer).
#' @param max_col_shift_down Maximum number of columns downwards (decreasing columnar indexes) that the projection can reach (default: 0, should be positive integer).
#' @param hem_shift Hemisphere shift for the projection: 0 = same hemisphere (default), 1 = contralateral hemisphere. Ignored when the network has only one hemisphere.
#' @return The updated motif object with the new projection loaded.
#' @export
load.projection.into.motif <- function(
    motif,
    presynaptic_layer,
    postsynaptic_layer,
    pre_neuron_fraction    = 0.5,
    presynaptic_type       = "principal",
    postsynaptic_type      = "principal",
    max_col_shift_up       = 0,
    max_col_shift_down     = 0,
    max_pch_shift_up       = 0,
    max_pch_shift_down     = 0, 
    hem_shift              = 0L,
    via_apical             = FALSE
  ) {
   
    # Checks
    if (length(presynaptic_layer) != 1) {stop("presynaptic_layer must be a single layer name.")}
    if (length(presynaptic_type)  != 1) {stop("presynaptic_type must be a single type name."  )}
    if (length(postsynaptic_type) != 1) {stop("postsynaptic_type must be a single type name." )}
    
    # Get length of postsynaptic_layer input
    n_post_layers     <- length(postsynaptic_layer)
    postsynaptic_type <- rep(postsynaptic_type, n_post_layers)
    
    # Set principal type by layer (only defined for cortical layers; subcortical layers require explicit type)
    if (presynaptic_type == "principal") {
      ptype <- principal.neurons()[[presynaptic_layer]]
      if (is.null(ptype)) stop(paste0("Cannot resolve 'principal' type for layer '", presynaptic_layer, "'. Specify presynaptic_type explicitly for subcortical layers."))
      presynaptic_type <- ptype
    }
    if (postsynaptic_type[1] == "principal") {
      for (i in c(1:n_post_layers)) {
        ptype <- principal.neurons()[[postsynaptic_layer[i]]]
        if (is.null(ptype)) stop(paste0("Cannot resolve 'principal' type for layer '", postsynaptic_layer[i], "'. Specify postsynaptic_type explicitly for subcortical layers."))
        postsynaptic_type[i] <- ptype
      }
    }
    
    # ... for each target layer
    for (i in c(1:n_post_layers)) {
      # Initialize new projection object
      proj <- new(Projection)
      # Load projection parameters 
      proj$pre_type   <- presynaptic_type
      proj$pre_layer  <- presynaptic_layer
      proj$post_type  <- postsynaptic_type[i]
      proj$post_layer <- postsynaptic_layer[i]
      proj$hem_shift  <- as.integer(hem_shift)
      proj$via_apical <- via_apical
      # Add projection to motif
      motif$load_projection(
        proj,
        as.integer(max_col_shift_up),
        as.integer(max_col_shift_down),
        as.integer(max_pch_shift_up),
        as.integer(max_pch_shift_down),
        pre_neuron_fraction
      )
    }
    
    return(motif)
    
  }

#' Set network structure
#' 
#' This function sets the structure of a network object, defining its layers, columns, neuron types, and local connectivity parameters. It also generates local nodes based on the specified structure.
#' 
#' @param network Network object to configure.
#' @param neuron_types Character vector giving types of neurons in the network. Known types can be accessed using \code{print.known.celltypes()}. Default is "principal", which will assign the most common neuron type for each layer, as defined in \code{principal.neurons()}.
#' @param layer_names Character vector giving names of cortical layers in the network, ordered deepest to most superficial, e.g. c("L6", "L5", "L4", "L3", "L2", "L1").
#' @param n_layers Integer giving number of cortical layers in the network.
#' @param n_columns Integer giving number of columns in the network.
#' @param n_patches Integer giving the number of "patches" (n_layers x n_columns sheets) in the network.
#' @param layer_height Numeric giving height of each layer (default value is 180.0, which assumes an implicit unit of micron).
#' @param column_diameter Numeric giving diameter of each column (default value is 120.0, which assumes an implicit unit of micron).
#' @param segment_length Numeric giving expected length of each segment in the axonal and dendritic processes of each neuron (default value is 20.0, which assumes an implicit unit of micron).
#' @param layer_separation_factor Numeric giving mean distance between layers as a fraction of layer height (default: 2.5).
#' @param column_separation_factor Numeric giving mean distance between columns as a fraction of column diameter (default: 2.5).
#' @param patch_separation_factor Numeric giving mean distance between network patches as a fraction of column diameter (default: 2.5). 
#' @param neurons_per_node Matrix giving mean number of neurons of each type per node in each layer, with cortical layers first and then subcortical layers; dimensions must match n_layers + n_subcortical_layers (rows) and length of neuron_types (columns), or 2 * (n_layers + n_subcortical_layers) if specifying different cell type counts for a second hemisphere. If there are two hemispheres but only n_layers + n_subcortical_layers rows, then the counts are reused for the second hemisphere. 
#' @param synaptic_neighborhood Numeric giving the radius (in microns) within which an axon node will trigger a synapse when near a dendrite node (default: 10.0).
#' @param n_hemispheres Integer giving number of hemispheres; must be 1 or 2 (default: 1).
#' @param hemisphere_names Character vector of length n_hemispheres giving names for the hemispheres (default: auto-generated as "left" or c("left","right")).
#' @param hem_separation_factor Numeric giving distance between hemispheres as a fraction of column diameter (default: 5.0).
#' @param n_subcortical_layers Integer giving number of subcortical layers (e.g., thalamic relay nuclei); can be 0 (default: 0).
#' @param subcortical_layer_names Character vector of length n_subcortical_layers giving names for the subcortical layers (default: auto-generated as "subL1", "subL2", ...). Must be distinct from all cortical layer names.
#' @param sub_separation_factor Numeric giving distance from the cortical sheet to the first subcortical layer as a fraction of layer height (default: 5.0).
#' @return The updated network object with the specified structure and local nodes generated.
#' @export
set.network.structure <- function(
    network,
    neuron_types                  = c("principal"),
    hemisphere_names              = NULL,
    subcortical_layer_names       = NULL,
    layer_names                   = c("layer"),
    n_hemispheres                 = 1,
    n_subcortical_layers          = 0,
    n_layers                      = 1,
    n_columns                     = 1,
    n_patches                     = 1,
    layer_height                  = 180.0,
    column_diameter               = 120.0,
    segment_length                = 20.0, 
    hem_separation_factor         = 40.0,
    sub_separation_factor         = 20.0,
    layer_separation_factor       = 2.5,
    column_separation_factor      = 2.5,
    patch_separation_factor       = 2.5,
    synaptic_neighborhood         = 10.0,
    neurons_per_node              = 10
  ) {
    
    # Check hemisphere count and auto-generate hemisphere names
    if (length(hemisphere_names) >= n_hemispheres && length(hemisphere_names) <= 2) {
      n_hemispheres <- length(hemisphere_names)
    }
    if (!(n_hemispheres %in% c(1L, 2L))) stop("n_hemispheres must be 1 or 2.")
    if (is.null(hemisphere_names)) {
      hemisphere_names <- if (n_hemispheres == 1) "left" else c("left", "right")
    }
    if (length(hemisphere_names) != n_hemispheres) {
      stop("Length of hemisphere_names must equal n_hemispheres.")
    }
    
    # Auto-generate subcortical layer names if needed
    if (n_subcortical_layers > 0 && is.null(subcortical_layer_names)) {
      subcortical_layer_names <- paste0("subL", seq_len(n_subcortical_layers))
    }
    if (n_subcortical_layers == 0) {
      if (length(subcortical_layer_names) > 0) {
        n_subcortical_layers <- length(subcortical_layer_names)
      } else {
        subcortical_layer_names <- character(0)
      }
    }
    if (length(subcortical_layer_names) != n_subcortical_layers) {
      stop("Length of subcortical_layer_names must equal n_subcortical_layers.")
    }
    
    # Check layer names
    if (length(layer_names) != n_layers) {
      if (n_layers > length(layer_names) && length(layer_names) == 1) {
        layer_names <- paste0(layer_names, "_", seq_len(n_layers))
      } else if (n_layers < length(layer_names) && n_layers == 1) {
        n_layers <- length(layer_names)
      } else {
        stop("Length of layer_names does not match n_layers, and neither is inferable from the other.")
      }
    }
    
    # Combine layer names
    layer_names_all <- c(layer_names, subcortical_layer_names)
    
    # Get correct number of rows for neurons_per_node 
    num_of_rows <- n_layers + n_subcortical_layers
    if (n_hemispheres == 2 && nrow(neurons_per_node) == 2 * num_of_rows) num_of_rows <- 2 * num_of_rows
   
    # Unpack "principal" neuron types 
    if ("principal" %in% neuron_types) {
      
      # ... remake neuron_types
      principals_by_layer <- sapply(layer_names_all, function(ln) principal.neurons()[[ln]])
      principals          <- unique(principals_by_layer)
      principal_idx       <- which(neuron_types == "principal")
      neuron_types        <- c(principals, neuron_types[-principal_idx])
      n_p                 <- length(principals)
      n_t                 <- length(neuron_types)
      nn_p_range          <- c(min(n_p + 1, n_t):n_t)
      n_types_old         <- n_t - n_p + 1
      
      # ... remake neurons_per_node
      neurons_per_node_new <- matrix(NA, nrow = num_of_rows, ncol = n_t)
      if (length(neurons_per_node) >= n_types_old) {
        if (!is.null(dim(neurons_per_node))) { # counts per layer and per type specified
          for (i in c(1:nrow(neurons_per_node))) {
            principal_counts <- c()
            for (t in seq_along(principals)) {
              if (principals[t] == principal.neurons()[[layer_names_all[i]]]) {
                principal_counts <- c(principal_counts, neurons_per_node[i, principal_idx])
              } else {
                principal_counts <- c(principal_counts, 0)
              }
            }
            neurons_per_node_new[i, ] <- c(principal_counts, neurons_per_node[i, -principal_idx])
          }
          neurons_per_node <- neurons_per_node_new
        } else { # only counts per type specified
          neurons_per_node <- c(rep(neurons_per_node[principal_idx], n_p), neurons_per_node[-principal_idx])
        }
      }
      
    }
   
    # Grab number of neuron types
    n_neuron_types <- length(neuron_types)
    
    # Check neuron counts per node
    if (!is.null(dim(neurons_per_node))) {
      npn_dim <- dim(neurons_per_node)
      if (any(npn_dim != c(num_of_rows, n_neuron_types))) {
        stop("Dimensions of neurons_per_node must match n_layers + n_subcortical_layers (or possibly 2x this if there are two hemispheres) and length of neuron_types.")
      }
    } else {
      if (length(neurons_per_node) == 1) {
        if (n_layers + n_subcortical_layers > 1) {
          neurons_per_node <- matrix(neurons_per_node, nrow = n_layers + n_subcortical_layers, ncol = n_neuron_types)
          npn_dim          <- dim(neurons_per_node)
        } else {
          neurons_per_node <- matrix(rep(neurons_per_node, n_neuron_types), nrow = 1, ncol = n_neuron_types)
          npn_dim          <- c(1, length(neurons_per_node))
        }
      } else if (length(neurons_per_node) == n_neuron_types) {
        neurons_per_node <- matrix(rep(neurons_per_node, n_layers + n_subcortical_layers), nrow = n_layers + n_subcortical_layers, ncol = n_neuron_types, byrow = TRUE)
        npn_dim          <- dim(neurons_per_node)
      } else {
        stop("Dimensions of neurons_per_node must match n_layers + n_subcortical_layers (or possibly 2x this if there are two hemispheres) and length of neuron_types, or be inferable from them.")
      }
    }
    
    # Helper: coerce conductance input to a list of n x n matrices
   
    # Set structure (new C++ signature: hsl_names as list, n as 5-vector, sep_factor as 5-vector)
    # Note: synaptic conductances are now automatically looked up from neuron type properties
    network$set_network_structure(
      neuron_types,
      list(hemisphere_names, subcortical_layer_names, layer_names),  # hsl_names: {hem, sub, lyr}
      as.integer(c(n_hemispheres, n_subcortical_layers, n_layers, n_columns, n_patches)),  # n
      c(hem_separation_factor, sub_separation_factor, layer_separation_factor, column_separation_factor, patch_separation_factor),  # sep_factors
      layer_height,
      column_diameter,
      segment_length, 
      synaptic_neighborhood,
      neurons_per_node
    )
    
    # Make local nodes and return
    network$make_local_nodes()
    return(network)
    
  }

#' Fetch network components
#' 
#' This function retrieves the components of a network object. 
#' @param network Network object from which to fetch components.
#' @param include_arbors Logical indicating whether to include arbor information in the fetched components (can be large and computationally intensive, default = FALSE).
#' @param return_arbors Logical indicating whether to return the raw arbor matrix in the output. When FALSE and include_arbors is TRUE, the arbor matrix is dropped to reduce memory usage (default: TRUE). The arbor matrix can be large (> 1 GB) and is used internally for computation but may not be needed by the user.
#' @param verbose Logical indicating whether to print a summary of the fetched components (default: TRUE).
#' @return A list containing the components of the network.
#' @export
fetch.network.components <- function(
    network, 
    include_arbors = FALSE,
    return_arbors  = TRUE,
    verbose        = TRUE
  ) {
    
    # Grab raw components
    network.components <- network$fetch_network_components(include_arbors)
    
    # Compute synapse distribution; synapse_counts is accumulated in C++ during the arbor-matrix fill pass
    if (include_arbors) {
      n_neurons <- network.components$n_neurons
      synapse_info <- data.frame(
        neuron_idx       = seq_len(n_neurons),
        neuron_type      = network.components$neuron_type_name,
        n_synapses       = network.components$synapse_counts,
        stringsAsFactors = FALSE
      )
      network.components$synapse_info <- synapse_info
      
      # Drop the raw arbor matrix if the caller doesn't need it.
      # The matrix is ~8 bytes * n_segments and can be > 1 GB for large networks;
      # omitting it keeps the returned object cacheable by knitr.
      if (!return_arbors) {
        network.components$arbors <- NULL
      }
    }
    
    # Print summary
    if (verbose) {
      cat("Summary of network:\n")
      cat("\tNumber of neurons:",            network.components$n_neurons,       "\n")
      if (include_arbors) {
        cat("\tNumber of synapses:", sum(    network.components$synapse_counts), "\n")
      }
      cat("\tHemisphere names:", paste(      network.components$hem_names,         collapse = ", "), "\n")
      cat("\tNumber of hemispheres:",        network.components$n_hem,           "\n")
      cat("\tSubortical layer names:", paste(network.components$sub_names,         collapse = ", "), "\n")
      cat("\tNumber of subcortical layers:", network.components$n_sub,           "\n")
      cat("\tCortical layer names:", paste(  network.components$layer_names,       collapse = ", "), "\n")
      cat("\tNumber of cortical layers:",    network.components$n_layers,        "\n")
      cat("\tNumber of columns:",            network.components$n_columns,       "\n")
      cat("\tNumber of patches:",            network.components$n_patches,       "\n")
      cat("\tCell types used:", paste(unique(network.components$neuron_type_name), collapse = ", "), "\n")
      if (network.components$n_neurons > 0) {
        cat("\tMotifs used:", paste(         network.components$edge_type_names,   collapse = ", "), "\n")
      } else {
        cat("\tMotifs used:\n")
      }
    }
    
    return(network.components)
    
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

# Internal helper: map a character vector of labels to a named color vector.
# Used by plot.network and plot.network.traces to share a consistent palette.
.network_label_colors <- function(labels) {
    known_label_colors <- list(
      "cell"               = "gray10",
      "subthreshold_only"  = "gray10",
      "thalamus"           = "gray20", 
      "layer"              = "gray50",
      "L1"                 = "gray50",
      "L2"                 = "lightskyblue3",
      "L2/3"               = "lightskyblue2",
      "L23"                = "lightskyblue2",
      "L3"                 = "lightskyblue1",
      "L4"                 = "slateblue1",
      "L5"                 = "skyblue1",
      "L6"                 = "royalblue1",
      "principal"          = "green3",
      "thalmacortical"     = "lightgreen", 
      "PN"                 = "green3", 
      "excitatory"         = "green3",
      "leaky_integrator"   = "green3",
      "slow_recovery"      = "green3", 
      "slow_drain"         = "green3", 
      "pyramidal"          = "green4",
      "callosal_pyramidal" = "darkolivegreen2",
      "pyramidal_L6"       = "green4",
      "spiny_stellate"     = "green2",
      "responsive spiny stellate" = "green2", 
      "interneuron"        = "red",
      "inhibitory"         = "red", 
      "bursting_cell"      = "red",
      "neurogliaform_cell" = "red", 
      "PV"                 = "violetred2",
      "retentive PV"       = "violetred2", 
      "callosal_PV"        = "palevioletred3",
      "SOM"                = "red3",
      "SST"                = "tomato",
      "VIP"                = "darkred",
      "axon"               = "green3",
      "dendrite"           = "darkred"
    )
    unknown_label_colors <- c("aquamarine1", "gray55", "gray75", "cyan", "cornflowerblue", "coral", "burlywood", "darkolivegreen")
    label_colors        <- rep("white", length(labels))
    names(label_colors) <- labels
    for (cl in seq_along(labels)) {
      label    <- labels[cl]
      hit_mask <- label == names(known_label_colors)
      if (any(hit_mask)) {
        label_colors[cl] <- known_label_colors[[which(hit_mask)[1]]]
      } else {
        label_colors[cl] <- sample(unknown_label_colors, 1)
      }
    }
    label_colors
  }

#' Plot network as directed graph
#' 
#' This function plots a network object as a directed graph using ggplot2. Nodes represent neurons, and directed edges represent connections between them. The plot can be customized by selecting which motif to display and how to color the edges.
#' 
#' @name plot.network
#' @rdname plot-network
#' @usage plot.network(
#'  network, 
#'  soma_mask = NULL,
#'  arbor_idx = NULL,
#'  threedim = FALSE,
#'  title = NULL, 
#'  soma_density = 1.0,
#'  arbor_density = 0.01,
#'  arbor_cell_type = "all",
#'  plot_motif = "all", 
#'  reconstruct_arbors = TRUE,
#'  edge_color = "pre_type", 
#'  soma_color = "layer", 
#'  soma_size_factor = 0.5, 
#'  return_plot = TRUE,
#'  return_cell_arbor_idx = TRUE
#' )
#' @param network Network object to plot.
#' @param soma_mask Logical vector of length equal to the number of neurons in the network, indicating which neurons to include in the plot (TRUE for included neurons, FALSE for excluded neurons). If NULL (default), a random sample of neurons will be selected based on the specified soma_density. Useful for reproducing the same cells across plots. 
#' @param arbor_idx Integer vector giving the indices of neurons for which to plot arbors (i.e., axonal and dendritic processes). If NULL (default), a random sample of neurons will be selected based on the specified arbor_density and arbor_cell_type. Useful for reproducing the same cells across plots. 
#' @param threedim Logical indicating whether to plot in 3D or to collapse the patch dimension and plot in 2D (default: FALSE).
#' @param title Title for the plot (default: "Cortex" or network name (if provided), plus plot motif name(s)).
#' @param soma_density Numeric value between 0 and 1 (inclusive) specifying what fraction of cell bodies are plotted (default: 1.0). Any value greater than 1 is treated as 1, any value less than or equal to 0 is treated as a call to plot a single cell body. 
#' @param arbor_density Numeric value between 0 and 1 (inclusive) specifying what fraction of cells (left after soma_density and cell-type restrictions are applied) have their axonal and dendritic arbors plotted, or, if not plotting arbors, their edges (default: 0.01). Any value greater than 1 is treated as 1, any value less than or equal to 0 is treated as a call to plot arbors for a single cell. Note that plotting arbors can be computationally intensive, so it is advisable to set this to a low value (e.g., 0.01) unless plotting edges only.
#' @param arbor_cell_type Character string specifying which cell type(s) to include when selecting cells for arbor plotting; options include "all" for all cell types, or the name (or character vector of names) of a specific cell type (default: "all").
#' @param plot_motif Character string specifying which motif to plot (applies only if plotting edges; as arbors can be built from multiple motifs, cannot plot arbors by motif); options include "all" for all, "local connections" for local connections within each node, or the name of a long-range projection motif (default: "all").
#' @param reconstruct_arbors Logical indicating whether to reconstruct axonal and dendritic arbors for the neurons in the plot, or whether to instead show synaptic connections as straight edges (default: TRUE, but can be computationally intensive).
#' @param edge_color Character string specifying how to color the edges; options include "pre_type" to color by presynaptic neuron type, "post_type" to color by postsynaptic neuron type, "motif" to color by motif type, and "is_axon" to color by whether a reconstructed arbor is an axon or dendrite (default: "pre_type"). Cannot use "post_type" or "motif" when reconstructing arbors, as arbor edges can be defined by multiple postsynaptic neuron types and motifs. Cannot use "is_axon" when not reconstructing arbors, as edges are not defined by axonal vs. dendritic processes.
#' @param soma_color Character string specifying how to color the nodes; options include "layer" to color by layer index or "type" to color by neuron type (default: "layer").
#' @param soma_size_factor Numeric value controlling how cell size in the plot scales to the number of cells (default: 0.5). 
#' @param return_plot Logical indicating whether to return the ggplot object or print the plot directly (default: TRUE).
#' @param return_cell_arbor_idx Logical indicating whether to return the soma_mask and arbor_idx used for plotting or not (default: TRUE).
#' @return Either prints the plot directly or returns the ggplot object, depending on the value of return_plot.
#' @export
plot.network <- function(
    network,
    soma_mask             = NULL,
    arbor_idx             = NULL,
    threedim              = FALSE,
    title                 = NULL,
    soma_density          = 1.0,
    arbor_density         = 0.01,
    arbor_cell_type       = "all",
    plot_motif            = "all",
    reconstruct_arbors    = TRUE,
    edge_color            = "pre_type",
    soma_color            = "layer",
    soma_size_factor      = 0.5,
    return_plot           = TRUE,
    return_cell_arbor_idx = TRUE
  ) {
    
    # Get network components
    ntw <- network$fetch_network_components(reconstruct_arbors) # Retrieve arbors? 
    units_distance <- "microns"
    
    # Check that soma_mask and arbor_idx are consistent 
    if (!is.null(soma_mask) && !is.null(arbor_idx)) {
      if (!all(arbor_idx %in% which(soma_mask))) {
        stop("All arbor_idx values must be included in soma_mask.")
      }
    }
    
    # Check edge and cell color
    if (!(edge_color %in% c("pre_type", "post_type", "motif", "is_axon"))) {
      stop("edge_color must be one of: 'pre_type', 'post_type', 'motif', or 'is_axon'.")
    }
    if (reconstruct_arbors) {
      if (edge_color == "motif") {
        stop("Cannot color by motif when reconstructing arbors, as arbors can be built from multiple motifs. Please choose a different edge_color option.")
      }
      if (edge_color == "post_type") {
        stop("Cannot color by postsynaptic neuron type when reconstructing arbors, as arbor edges are defined by presynaptic neuron. Please choose a different edge_color option.")
      }
    } else {
      if (edge_color == "is_axon") {
        stop("Cannot color by axon vs. dendrite when not reconstructing arbors, as edges are not defined by axonal vs. dendritic processes. Please choose a different edge_color option.")
      }
    }
    if (!(soma_color %in% c("layer", "type"))) {
      stop("soma_color must be one of: 'layer' or 'type'.")
    }
   
    # Set plot title 
    if (is.null(title)) {
      title <- "Network Topology"
    }
   
    # Find range of possible neurons for arbor plotting, based on cell type if specified
    celltype_mask <- rep(TRUE,  ntw$n_neurons)
    if (arbor_cell_type != "all") {
      celltype_mask <- rep(FALSE,  ntw$n_neurons) 
      for (act in arbor_cell_type) celltype_mask <- celltype_mask | ntw$neuron_type_name == act
      if (!is.null(soma_mask)) celltype_mask <- celltype_mask & soma_mask
      if (sum(celltype_mask) == 0) stop("No neurons match the specified arbor_cell_type")
    }
    motif_mask <- rep(TRUE,  ntw$n_neurons)
    if (plot_motif != "all") {
      motif_mask <- rep(FALSE,  ntw$n_neurons)
      for (mot in plot_motif) {
        if (sum(mot == colnames(ntw$arbor_motifs)) == 0) stop("Requested motif type is not in the network")
        motif_mask <- motif_mask | ntw$arbor_motifs[,mot] == 1
      }
      if (!is.null(soma_mask)) motif_mask <- motif_mask & soma_mask
      if (sum(motif_mask) == 0) stop("No neurons match the specified plot_motif")
    }
    celltype_motif_mask <- celltype_mask & motif_mask 
    if (sum(celltype_motif_mask) == 0) stop("No neurons match both the specified arbor_cell_type and plot_motif")
    
    if (is.null(soma_mask)) {
      # Get number of cell bodies to plot 
      n_soma <- 1
      if (soma_density > 0) n_soma <- round(ntw$n_neurons * min(1, soma_density))
      # Make mask for soma
      soma_idx            <- sort(sample(ntw$n_neurons, n_soma, replace = FALSE))
      soma_mask           <- celltype_motif_mask
      soma_mask[soma_idx] <- TRUE
      n_soma              <- sum(soma_mask)
    } else {
      soma_mask <- soma_mask & celltype_motif_mask
      n_soma    <- sum(celltype_motif_mask)
    }
    
    # Randomly select neurons for arbor plotting from among those that are in the soma plot and match the specified cell type (if applicable)
    if (is.null(arbor_idx)) {
      n_arbors <- 1
      if (arbor_density > 0) n_arbors <- round(sum(soma_mask) * min(1, arbor_density)) 
      arbor_idx             <- sort(sample(which(celltype_motif_mask), n_arbors, replace = FALSE))
    } else {
      arbor_mask            <- rep(FALSE, length(soma_mask))
      arbor_mask[arbor_idx] <- TRUE 
      arbor_mask            <- arbor_mask & soma_mask
      arbor_idx             <- which(arbor_mask)
    }
    
    # Get layer information - resolve each neuron to a display name
    # coords_node columns: hem_idx, sub_lyr_idx, patch_idx, lyr_idx, col_idx
    # -1 in lyr_idx means subcortical; -1 in sub_lyr_idx means cortical
    node_tbl    <- ntw$coordinates_node[soma_mask, , drop = FALSE]
    lyr_idx_vec <- node_tbl[, "lyr_idx"]      # 1-based cortical layer, or -1
    sub_idx_vec <- node_tbl[, "sub_lyr_idx"]  # 1-based subcortical layer, or -1
    neuron_layer_labels <- character(nrow(node_tbl))
    for (.i in seq_len(nrow(node_tbl))) {
      if (lyr_idx_vec[.i] >= 1) {
        neuron_layer_labels[.i] <- ntw$layer_names[lyr_idx_vec[.i]]
      } else if (sub_idx_vec[.i] >= 1) {
        neuron_layer_labels[.i] <- ntw$sub_names[sub_idx_vec[.i]]
      } else {
        neuron_layer_labels[.i] <- "unknown"
      }
    }
    
    # Create cells dataframe
    y_coord <- "y"
    z_coord <- "z"
    if (threedim) {
      y_coord <- "z"
      z_coord <- "y"
    }
    soma <- data.frame(
      idx   = c(1:n_soma), 
      x     = ntw$coordinates_spatial[soma_mask, "x"], 
      y     = ntw$coordinates_spatial[soma_mask, y_coord],
      z     = ntw$coordinates_spatial[soma_mask, z_coord],
      layer = as.factor(neuron_layer_labels),
      type  = ntw$neuron_type_name[soma_mask]
    )
    
    # Get cell edge pairs / reconstruct arbors
    synapses_included <- FALSE
    if (reconstruct_arbors) {
      
      # Reconstruct the edge matrix for these cells only
      edges <- ntw[["arbors"]]
      
      # Find the number of edges in reconstruction
      n_downsample_edges <- c()
      for (n in arbor_idx) {
        n_edges            <- sum(edges[,"neuron_idx"] == n)
        n_downsample_edges <- c(n_downsample_edges, n_edges)
      }
      
      # Make downsampled matrix
      edges_downsampled <- matrix(NA, nrow = sum(n_downsample_edges), ncol = ncol(edges))
      idx_start         <- 1
      idx_end           <- 0
      for (i in seq_along(arbor_idx)) {
        idx_start <- idx_end + 1
        idx_end   <- idx_end + n_downsample_edges[i]
        edges_downsampled[idx_start:idx_end, ] <- edges[edges[,"neuron_idx"] == arbor_idx[i], ]
      }
      
      # Add colnames: "neuron_idx", "arbor_id", "is_axon", "node_type", "parent_idx", "is_leaf", "is_synapse", "z_start", "y_start", "x_start", "z_end", "y_end", "x_end"
      # ... note: "neuron_idx" is for the pre-synaptic cell, i.e. = "pre_idx" below. 
      colnames(edges_downsampled) <- colnames(edges)
      
      # Make into data frame and rename axons and node type
      edges_downsampled <- as.data.frame(edges_downsampled) 
      edges_downsampled$is_axon[edges_downsampled$is_axon == 1]     <- "axon"
      edges_downsampled$is_axon[edges_downsampled$is_axon == 0]     <- "dendrite"
      edges_downsampled$node_type[edges_downsampled$node_type == 0] <- "soma"
      edges_downsampled$node_type[edges_downsampled$node_type == 1] <- "dendrite_shaft"
      edges_downsampled$node_type[edges_downsampled$node_type == 2] <- "axon_shaft"
      edges_downsampled$node_type[edges_downsampled$node_type == 3] <- "spine"
      
      # Find segment lengths
      edges_downsampled$seg_length <- sqrt(
        (edges_downsampled[,"x_end"] - edges_downsampled[,"x_start"])^2 +
        (edges_downsampled[,"y_end"] - edges_downsampled[,"y_start"])^2 +
        (edges_downsampled[,"z_end"] - edges_downsampled[,"z_start"])^2
      )
      
      # Add cell type 
      edges_downsampled$pre_type <- ntw$neuron_type_name[edges_downsampled$neuron_idx]
      
      # Rename the "neuron_idx" column to "pre_idx" to match the edge dataframe format used for non-arbor plotting
      colnames(edges_downsampled)[colnames(edges_downsampled) == "neuron_idx"] <- "pre_idx" 
      
      # Get synapses, if any
      synapses_included <- any(edges_downsampled$is_synapse > 0)
      if (synapses_included) {
        synapse_coordinates <- edges_downsampled[edges_downsampled$is_synapse > 0, c("z_end", "y_end", "x_end")]
        colnames(synapse_coordinates) <- c("z", "y", "x")
      }
      
      # Update edges
      edges <- edges_downsampled
      rm(edges_downsampled)
      
    } else {
      
      # Getting motifs for plotting
      edge_type_names <- ntw$edge_type_names
      et_masked       <- seq_along(edge_type_names)
      if (plot_motif != "all") {
        edge_type_mask  <- edge_type_names %in% plot_motif
        edge_type_names <- edge_type_names[edge_type_mask]
        et_masked       <- which(edge_type_mask)
        if (sum(edge_type_mask) == 0) stop("No edge types match the specified plot_motif.")
      }
      
      # Collect edges by motifs
      edges <- matrix(0, nrow = 0, ncol = 5)
      for (et in seq_along(edge_type_names)) {
        et_name  <- edge_type_names[et]
        et_edges <- ntw$edge_idx_by_type[[et_masked[et]]]
        for (ni in unique(et_edges[, "pre_neuron_idx"])) {
          if (sum(ni == arbor_idx) == 0) {
            ni_mask  <- et_edges[, "pre_neuron_idx"] != ni
            et_edges <- et_edges[ni_mask, ]
          }
        }
        et_edges <- cbind(
          et_edges, 
          rep(et_name, nrow(et_edges)),
          ntw$neuron_type_name[et_edges[,"pre_neuron_idx"]],
          ntw$neuron_type_name[et_edges[,"post_neuron_idx"]]
        )
        edges <- rbind(edges, et_edges)
      }
      edges <- as.data.frame(edges)
      colnames(edges) <- c("pre_idx", "post_idx", "motif", "pre_type", "post_type")
      
      # Find coordinates for start and end of edges
      edges$x_start <- soma[edges$pre_idx, "x"]
      edges$y_start <- soma[edges$pre_idx, "y"]
      edges$z_start <- soma[edges$pre_idx, "z"]
      edges$x_end   <- soma[edges$post_idx, "x"]
      edges$y_end   <- soma[edges$post_idx, "y"]
      edges$z_end   <- soma[edges$post_idx, "z"]
      
    }
    
    # Set point size to scale with number of cells
    if (threedim) soma_size_factor <- soma_size_factor * 2
    soma_size <- soma_size_factor * 10 / log(n_soma + 1)
    
    # Make colors 
    if (length(unique(as.character(soma[,soma_color]))) == 1) soma[,soma_color] <- "cell"
    colored_labels <- unique(
      c(unique(as.character(edges[,edge_color])), 
        unique(as.character(soma[,soma_color])))
      )
    label_colors <- .network_label_colors(colored_labels)
    
    # Add color for synapses
    syn_color <- "orange"
    if (synapses_included) {
      label_colors <- c(label_colors, syn_color)
      names(label_colors)[length(label_colors)] <- "synapse"
    }
    
    # Plot
    title_size  <- 14 
    axis_size   <- 12 
    legend_size <- 10
    
    if (threedim) {
      
      # Translate colors for plotly
      hex <- rgb(t(col2rgb(label_colors)), maxColorValue = 255)
      
      # Remake level names for plotly
      if (edge_color == "is_axon") {
        level_names <- c("axon", "dendrite", ntw$layer_names, ntw$sub_names)
      } else if (edge_color == "pre_type") {
        level_names <- c(unique(edges$pre_type), ntw$layer_names, ntw$sub_names)
      } else {
        stop("edge_color must be 'is_axon' or 'pre_type' when reconstructing arbors.")
      }
      if (synapses_included) {
        level_names <- c(level_names, "synapse")
        # ... and reset levels in synapse_coordinates
        synapse_coordinates$synapse <- "synapse" 
        synapse_coordinates$synapse <- factor(synapse_coordinates$synapse, levels = level_names, labels = level_names)
      }
      
      # Reset levels in soma and edge data frames
      soma$layer         <- factor(soma$layer, levels = level_names, labels = level_names)
      edges[,edge_color] <- factor(edges[,edge_color], levels = level_names, labels = level_names) 
      
      # Make long version of edges for faster ploting in plotly
      edges_long <- data.frame(
        x     = c(rbind(edges$x_start, edges$x_end, NA)),
        y     = c(rbind(edges$y_start, edges$y_end, NA)),
        z     = c(rbind(edges$z_start, edges$z_end, NA)),
        group = rep(edges[[edge_color]], each = 3)
      )
      
      # Initialize plotly plot with edges
      plt <- plotly::plot_ly(
        edges_long,
        x      = ~x,
        y      = ~z,
        z      = ~y,
        type   = "scatter3d",
        mode   = "lines",
        color  = ~factor(group),
        colors = hex
      )
      
      # Add soma as points
      plt <- plt |>
        plotly::add_trace(
          data   = soma,
          x      = ~x,
          y      = ~y,
          z      = ~z,
          type   = "scatter3d",
          mode   = "markers",
          marker = list(size = soma_size),
          color  = ~factor(layer),
          colors = hex
        ) 
      
      # Add synapses
      if (synapses_included) {
        plt <- plt |> 
          plotly::add_trace(
            data   = synapse_coordinates,
            x      = ~x,
            y      = ~z,
            z      = ~y,
            type   = "scatter3d",
            mode   = "markers",
            marker = list(size = soma_size/2),
            color  = ~synapse,
            colors = hex
          )
      }
      
      # Label axes and fix colors into light mode 
      plt <- plt |>
        plotly::layout(
          template      = "plotly_white",
          paper_bgcolor = "white",
          plot_bgcolor  = "white",
          font          = list(color = "black"),
          scene         = list(
            xaxis = list(title = "Cortical Columns", color = "black", backgroundcolor = "white"),
            zaxis = list(title = "Cortical Layers", color = "black", backgroundcolor = "white"),
            yaxis = list(title = "Cortical Patches", color = "black", backgroundcolor = "white")
          )
        )
      
      # Evaluate lazy closures and strip the visdat/attrs environment references.
      # plotly stores trace data as closures (in $x$visdat) that capture the
      # imports:plotly environment; this causes raw serialized size to exceed
      # R's 2^31-byte limit for rawConnection(), breaking knitr caching.
      # plotly_build() evaluates those closures into $x$data; stripping the
      # leftover closures and the preRenderHook (which re-invokes them) keeps
      # the object serializable while leaving the rendered output unchanged.
      plt <- plotly::plotly_build(plt)
      plt$x$visdat      <- NULL
      plt$x$attrs       <- NULL
      plt$preRenderHook <- NULL
      
    } else {
      
      plt <- ggplot2::ggplot() +
        # soma as points
        ggplot2::geom_point(data = soma, size = soma_size, ggplot2::aes(x = x, y = y, color = .data[[soma_color]])) +
        # edges as arrows
        ggplot2::geom_segment(
          data = edges,
          ggplot2::aes(x = x_start, y = y_start, xend = x_end, yend = y_end, color = .data[[edge_color]])
        ) +
        ggplot2::theme_minimal() +
        ggplot2::labs(
          title = title, 
          x     = paste0("columnar coordinate (", units_distance, ")"), 
          y     = paste0("laminar coordinate (", units_distance, ")")
        ) + 
        ggplot2::scale_colour_manual(
          name   = "Types",
          values = label_colors
        ) +
        ggplot2::guides(color = ggplot2::guide_legend(override.aes = list(alpha = 1))) +
        ggplot2::theme(
          panel.background = ggplot2::element_rect(fill = "white", colour = NA),
          plot.background  = ggplot2::element_rect(fill = "white", colour = NA),
          plot.title       = ggplot2::element_text(hjust = 0.5, size = title_size),
          axis.title       = ggplot2::element_text(size = axis_size),
          axis.text        = ggplot2::element_text(size = axis_size),
          legend.title     = ggplot2::element_text(size = legend_size),
          legend.text      = ggplot2::element_text(size = legend_size) 
        )
      
      # Add synapses 
      if (synapses_included) {
        plt <- plt + 
          ggplot2::geom_point(
            data = synapse_coordinates,
            ggplot2::aes(x = x, y = y, color = "synapse"),
            size = soma_size/2
          )
      }
      
    }
    
    if (return_plot) {
      if (return_cell_arbor_idx) {
        return(list(plot = plt, soma_mask = soma_mask, arbor_idx = arbor_idx))
      } else {
        return(list(plot = plt))
      }
    } else {
      print(plt)
      if (return_cell_arbor_idx) {
        return(list(soma_mask = soma_mask, arbor_idx = arbor_idx))
      } else {
        return(invisible(NULL))
      }
    }
    
  }

#' Plot spike traces for network from SGT simulation 
#' 
#' This function plots spike traces for a network object from a Spatial Growth-Transform (SGT) simulation. 
#' 
#' @name plot.network.traces
#' @rdname plot-network-traces
#' @usage plot.network.traces(network, return_plot = FALSE, input_matrix = NULL, window_size = 0.01)
#' @param network Network object with SGT simulation traces to plot.
#' @param return_plot Logical indicating whether to return the ggplot object (TRUE) or print it (FALSE) (default: FALSE).
#' @param input_matrix Matrix of stimulus currents, with rows representing neurons and columns representing sample times. Presumably the one used to generate the traces. Options. If provided, will be added to the bottom of the plot. 
#' @param window_size Proportion of time steps to use as a moving window for computing spike rate (default: 0.01). 
#' @return A ggplot object showing spike traces for all neurons in the network over time.
#' @export
plot.network.traces <- function(
    network,
    return_plot  = FALSE,
    input_matrix = NULL,
    window_size  = 0.01,
    plot_rates   = TRUE
  ) {
    
    # Get simulation results and network components
    v_traces    <- network$fetch_sim_results()$v_traces
    ntw         <- network$fetch_network_components(FALSE)
    n_neurons   <- nrow(v_traces)
    n_time      <- ncol(v_traces)
    print_input <- !is.null(input_matrix)
    
    if (print_input) {
      if (!all(dim(input_matrix) == dim(v_traces))) {
        stop("input matrix and v_traces differ in dimensions")
      }
    }
    
    # Make time vector
    time_seq <- seq(1, by = ntw$sim_dt, length.out = n_time)
    
    # Detect spikes before downsampling
    jump_threshold <- 20  # mV
    spike_matrix   <- matrix(FALSE, nrow = n_neurons, ncol = n_time)
    if (n_time > 1) {
      col_diffs          <- v_traces[, -1, drop = FALSE] - v_traces[, -n_time, drop = FALSE]
      spike_matrix[, -1] <- col_diffs > jump_threshold
    }
    
    # Calculate moving window size
    total_duration  <- max(time_seq) - min(time_seq)
    window_duration <- window_size * total_duration
    window_samples  <- max(1L, round(window_duration / ntw$sim_dt))
    if (window_samples %% 2 == 0) {window_samples <- window_samples + 1L}
    half_window     <- floor(window_samples / 2)
    
    # Get cell type information
    neuron_type <- ntw$neuron_type_name
    cell_types  <- unique(neuron_type)
    
    # Calculate population-average firing rate for each type
    spike_rate <- matrix(
      NA_real_,
      nrow     = length(cell_types),
      ncol     = n_time,
      dimnames = list(cell_types, NULL)
    )
    
    for (k in seq_along(cell_types)) {
     
      # Sum spikes across neurons of this type
      idx <- which(neuron_type == cell_types[k])
      population_spikes <- colSums(spike_matrix[idx, , drop = FALSE])
      n_type_neurons <- length(idx)
      
      # Calculate centered moving sum using cumulative sums
      cs         <- c(0, cumsum(population_spikes))
      moving_sum <- rep(NA_real_, n_time)
      centers    <- (half_window + 1):(n_time - half_window)
      
      if (length(centers) > 0) {
        left_edges          <- centers - half_window
        right_edges         <- centers + half_window
        moving_sum[centers] <- cs[right_edges + 1] - cs[left_edges]
      }
      
      # Convert spikes/window to Hz/neuron
      spike_rate[k, ] <- moving_sum / (window_samples * ntw$sim_dt / 1000) / n_type_neurons
    }
    
    # Downsample for plotting
    keep_cols <- seq_len(n_time)
    if (length(v_traces) > 1e6) {
      target_cols <- floor(1e6 / nrow(v_traces))
      
      # Columns containing at least one spike
      spike_cols <- which(apply(spike_matrix, 2, any))
      n_spike    <- length(spike_cols)
      
      # Non-spike columns 
      n_fill         <- max(target_cols - n_spike, 1000)
      non_spike_cols <- setdiff(seq_len(n_time), spike_cols)
      fill_cols      <- if (n_fill >= length(non_spike_cols)) {
        non_spike_cols
      } else {
        non_spike_cols[round(seq(1, length(non_spike_cols), length.out = n_fill))]
      }
      
      # Set columns to keep
      keep_cols <- sort(unique(c(1, fill_cols, spike_cols, n_time)))
      
      # Downsample
      v_traces   <- v_traces[, keep_cols, drop = FALSE]
      time_seq   <- time_seq[keep_cols]
      spike_rate <- spike_rate[, keep_cols, drop = FALSE]
      if (print_input) {
        input_matrix <- input_matrix[, keep_cols, drop = FALSE]
      }
    }
    
    # Input matrix if none supplied
    if (!print_input) {
      input_matrix <- matrix(NA_real_, nrow = nrow(v_traces), ncol = ncol(v_traces))
    } 
    
    # Long-format trace data
    v_traces_long <- data.frame(
      time      = rep(time_seq, times = n_neurons),
      potential = as.vector(t(v_traces)),
      id        = rep(seq_len(n_neurons), each = length(time_seq)),
      type      = rep(neuron_type, each = length(time_seq)),
      input     = as.vector(t(input_matrix))
    )
    v_traces_long$id <- as.character(v_traces_long$id)
    label_colors     <- .network_label_colors(cell_types)
    
    # Construct one patchwork block per cell type
    title_size <- 10
    axis_size  <- 6
    
    plots_by_type <- lapply(seq_along(cell_types), 
      function(k) {
        
        cell_type <- cell_types[k]
        trace_data <- v_traces_long[v_traces_long$type == cell_type, , drop = FALSE]
        rate_data <- data.frame(time = time_seq, spike_rate = spike_rate[k, ])
        
        # Spike-rate plot
        plt_rate <- ggplot2::ggplot(
          rate_data,
          ggplot2::aes(
            x = time,
            y = spike_rate
          )
        ) +
          ggplot2::geom_line(linewidth = 0.7) +
          ggplot2::theme_minimal() +
          ggplot2::theme(
            panel.background = ggplot2::element_rect(fill = "white", colour = NA),
            plot.background  = ggplot2::element_rect(fill = "white", colour = NA),
            plot.title       = ggplot2::element_text(hjust = 0.5, size = title_size),
            axis.title.x     = ggplot2::element_blank(),
            axis.text.x      = ggplot2::element_blank(),
            axis.ticks.x     = ggplot2::element_blank(),
            axis.title.y     = ggplot2::element_text(size = axis_size),
            axis.text.y      = ggplot2::element_text(size = axis_size)
          ) +
          ggplot2::labs(
            title = cell_type,
            y = "Spike rate (Hz)"
          )
        
        # Membrane-potential traces
        plt_trace <- ggplot2::ggplot(
          trace_data,
          ggplot2::aes(
            x = time,
            y = potential,
            group = id
          )
        ) +
          ggplot2::geom_line(
            linewidth = 0.5,
            color = label_colors[k]
          ) +
          ggplot2::theme_minimal() +
          ggplot2::theme(
            panel.background = ggplot2::element_rect(fill = "white", colour = NA),
            plot.background  = ggplot2::element_rect(fill = "white", colour = NA),
            plot.title       = ggplot2::element_text(hjust = 0.5, size = title_size),
            axis.title.x     = ggplot2::element_blank(),
            axis.text.x      = ggplot2::element_blank(),
            axis.ticks.x     = ggplot2::element_blank(),
            axis.title.y     = ggplot2::element_text(size = axis_size),
            axis.text.y      = ggplot2::element_text(size = axis_size)
          ) +
          ggplot2::labs(
            y = "potential (mV)"
          )
        
        # Input-current plot
        if (print_input) {
          # Remove x axis from plot trace 
          plt_trace <- plt_trace + ggplot2::theme(
            axis.title.x     = ggplot2::element_blank(),
            axis.text.x      = ggplot2::element_blank(),
            axis.ticks.x     = ggplot2::element_blank()
          )
         
          # Make input plot
          plt_input <- ggplot2::ggplot(
            trace_data,
            ggplot2::aes(
              x = time,
              y = input,
              group = id
            )
          ) +
            ggplot2::geom_line(
              linewidth = 0.6,
              color = label_colors[k]
            ) +
            ggplot2::theme_minimal() +
            ggplot2::theme(
              panel.background = ggplot2::element_rect(fill = "white", colour = NA),
              plot.background  = ggplot2::element_rect(fill = "white", colour = NA),
              plot.title       = ggplot2::element_text(hjust = 0.5, size = title_size),
              axis.title.y     = ggplot2::element_text(size = axis_size),
              axis.text.y      = ggplot2::element_text(size = axis_size)
            ) +
            ggplot2::labs(
              x = "Time (ms)",
              y = "Input (pA)"
            )
          
          if (plot_rates) {
            patchwork::wrap_plots(
              plt_rate,
              plt_trace,
              plt_input,
              ncol    = 1,
              heights = c(2, 2, 1)
            )
          } else {
            patchwork::wrap_plots(
              plt_trace,
              plt_input,
              ncol    = 1,
              heights = c(2, 1)
            )
          }
          
        } else {
          
          if (plot_rates) {
            patchwork::wrap_plots(
              plt_trace,
              ncol    = 1,
              heights = c(1)
            )
          } else {
            patchwork::wrap_plots(
              plt_rate,
              plt_trace,
              ncol    = 1,
              heights = c(1, 1)
            )
          }
          
        }
      }
    )
    
    # Stack all cell types
    plt <- patchwork::wrap_plots(plots_by_type, ncol = 1)
    
    if (return_plot) {
      return(plt)
    } else {
      print(plt)
      return(invisible(NULL))
    }
  }

#' Run Spatial Growth-Transform network simulation
#' 
#' This function uses a Spatial Growth-Transform (SGT) model to run a spike simulation on a given network object for a specified matrix of membrane currents over time. A matrix containing the spike traces of all neurons over time after the simulation (neurons as rows, sample times as columns) is saved in the network object, along with a vector of spike counts for each neuron in the network. Both are returned on the R side in a list.
#' 
#' @param network Network object on which to run the simulation.
#' @param stimulus_current_matrix Matrix of stimulus currents, with rows representing neurons and columns representing sample times.
#' @param dt Time step length in ms (default: 1e-3, i.e., 1 micosecond time steps).
#' @param initial_potential Initial value for membrane potential, applied to all cells (default is -70 mV).
#' @return List containing the following elements: \item{v_traces}{Matrix of simulated subthreshold voltage + spike traces for all neurons over time (neurons as rows, sample times as columns).} \item{spike_counts}{Vector of spike counts for each neuron in the network.} 
#' @export
run.SGT <- function(
    network,
    stimulus_current_matrix, 
    dt                = 1e-3,  
    initial_potential = -70.0
  ) {
    network$SGT(stimulus_current_matrix, dt, initial_potential)
    return(network$fetch_sim_results())
  }
