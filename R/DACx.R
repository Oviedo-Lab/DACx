
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
    Rcpp::loadModule("motif", TRUE)
    Rcpp::loadModule("network", TRUE)
    Rcpp::loadModule("Projection", TRUE)
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
#' This function adds a user-defined cell type to the current session. It's just a wrapper for the Rcpp-exported \code{add_cell_type} function. Technically, \code{cell_type} is a struct defined in the Rcpp backend of the DACx package. They are essentially labeled lists whose fields are described by the parameters below. Each session stores cell types in the Rcpp backend in an \code{unordered_map} with string labels. All parameters come with biologically realistic (and mathematically workable) default values, except for \code{type_name} and \code{valence}. 
#' 
#' @param type_name Character string giving name of the cell type, e.g. "pyramidal", "PV", "SST", etc.
#' @param valence Valence of each neuron type, +1 for excitatory, -1 for inhibitory.
#' @param tau_fast Placeholder
#' @param tau_slow Placeholder
#' @param tau_Vs Placeholder
#' @param I_slow Placeholder 
#' @param U_Vs Placeholder 
#' @param max_spike_rate Number of spikes which can be "cleared" per ms (spikes/ms). Default is 0.01.
#' @param transmission_velocity Transmission velocity (in microns/ms) for each neuron type. Default value is 30e3.
#' @param spine_density Scale between 0 and 1; 0 = no spines, 1 = every node along dendrite is a spine. Default is 0.0. 
#' @param axon_target Character string giving target of axon projections for each neuron type, one of: "spine", "dendrite_shaft", "soma", or "axon_shaft". Default is "dendrite_shaft".
#' @param I_spike Spike current, in pA. Default value is 1e3 (i.e., 1 nA); absolute value (plus a little bit) used as \code{dHdv_bound}.
#' @param spike_potential Magnitude of each spike, in mV. Default value is 35.0.
#' @param resting_potential Resting potential, in mV. Default value is -70.0; absolute value (plus a little bit) used as \code{v_bound}.
#' @param threshold Spike threshold, in mV. Default value is -55.0.
#' @param leak_conductance Conductance controlling the leak current, \code{I_leak = leak_conductance (resting_potential - v)}, in nS. Default value is 10 nS.
#' @param axon_branch_count Expected number of axon branches. Default is 10. 
#' @param dendrite_branch_count Expected number of dendrite branches. Default is 10. 
#' @param branch_independence Scale between 0 and 1; 0 = all branches connect to soma from single segment, 1 = all branches connect directly to soma. Default is 0.5.
#' @param branch_spread Scale between 0 and 1; 0 = no tendency to extend away from soma, 1 = straight line away from soma. Default is 0.5.
#' @param apical_target_layer Character string giving target layer for apical dendrites. Default: "none".
#' @return Nothing.
#' @export
add.cell.type <- function(
    type_name,
    valence,
    tau_fast                         = 1.0,
    tau_slow                         = 60.0,
    tau_Vs                           = 100.0,
    I_slow                           = 0.01, 
    U_Vs                             = 0.05, 
    max_spike_rate                   = 0.1,
    transmission_velocity            = 30e3,
    spine_density                    = 0.0,
    axon_target                      = "dendrite_shaft",
    I_spike                          = 1e3,
    spike_potential                  = 35.0,
    resting_potential                = -70.0,
    threshold                        = -55.0,
    leak_conductance                 = 10.0,
    axon_branch_count                = 10L,
    dendrite_branch_count            = 10L,
    branch_independence              = 0.5,
    branch_spread                    = 0.5,
    apical_target_layer              = "none"
  ) {
    add_cell_type(list(
      type_name                        = type_name,
      valence                          = as.integer(valence),
      tau_fast                         = tau_fast, 
      tau_slow                         = tau_slow, 
      tau_Vs                           = tau_Vs, 
      I_slow                           = I_slow, 
      U_Vs                             = U_Vs, 
      max_spike_rate                   = max_spike_rate,
      transmission_velocity            = transmission_velocity,
      spine_density                    = spine_density,
      axon_target                      = axon_target,
      I_spike                          = I_spike,
      spike_potential                  = spike_potential,
      resting_potential                = resting_potential,
      threshold                        = threshold,
      leak_conductance                 = leak_conductance,
      axon_branch_count                = as.integer(axon_branch_count),
      dendrite_branch_count            = as.integer(dendrite_branch_count),
      branch_independence              = branch_independence,
      branch_spread                    = branch_spread,
      apical_target_layer              = apical_target_layer
    ))
  }

#' Modify existing cell type 
#' 
#' This function modifies parameters of an existing cell type in the current session. Parameters can be updated selectively. If a parameter is not specified (or is specified as \code{NULL}), the existing value will be kept.
#' 
#' @param type_name Character string giving name of the cell type, e.g. "excitatory", "inhibitory", "PV", "SST", etc.
#' @param valence Valence of each neuron type, +1 for excitatory, -1 for inhibitory.
#' @param tau_fast Placeholder
#' @param tau_slow Placeholder
#' @param tau_Vs Placeholder
#' @param I_slow Placeholder 
#' @param U_Vs Placeholder 
#' @param max_spike_rate Number of spikes which can be "cleared" per ms (spikes/ms).
#' @param transmission_velocity Transmission velocity (in microns/ms) for each neuron type.
#' @param spine_density Scale between 0 and 1; 0 = no spines, 1 = every node along dendrite is a spine.
#' @param axon_target Character string giving target of axon projections, one of: "spine", "dendrite_shaft", "soma", or "axon_shaft".
#' @param I_spike Spike current, in pA; absolute value (plus a little bit) used as \code{dHdv_bound}.
#' @param spike_potential Magnitude of each spike, in mV.
#' @param resting_potential Resting potential, in mV; absolute value (plus a little bit) used as \code{v_bound}.
#' @param threshold Spike threshold, in mV.
#' @param leak_conductance Conductance controlling the leak current, \code{I_leak = leak_conductance (resting_potential - v)}, in nS.
#' @param axon_branch_count Expected number of axon branches.
#' @param dendrite_branch_count Expected number of dendrite branches.
#' @param branch_independence Scale between 0 and 1; 0 = all branches connect to soma from single segment, 1 = all branches connect directly to soma.
#' @param branch_spread Scale between 0 and 1; 0 = no tendency to extend away from soma, 1 = straight line away from soma.
#' @param apical_target_layer Character string giving target layer for apical dendrites.
#' @return Nothing.
#' @export
modify.cell.type <- function(
    type_name,
    valence                          = NULL,
    tau_fast         = NULL,
    tau_slow = NULL,
    tau_Vs    = NULL,
    I_slow = NULL, 
    U_Vs = NULL, 
    max_spike_rate              = NULL,
    transmission_velocity            = NULL,
    spine_density                    = NULL,
    axon_target                      = NULL,
    I_spike                          = NULL,
    spike_potential                  = NULL,
    resting_potential                = NULL,
    threshold                        = NULL,
    leak_conductance                 = NULL,
    axon_branch_count                = NULL,
    dendrite_branch_count            = NULL,
    branch_independence              = NULL,
    branch_spread                    = NULL,
    apical_target_layer              = NULL
  ) {
    ep <- fetch.cell.type.params(type_name)  # existing params
    if (is.null(valence))                          valence                          <- ep$valence
    if (is.null(tau_fast))         tau_fast         <- ep$tau_fast
    if (is.null(tau_slow)) tau_slow <- ep$tau_slow
    if (is.null(tau_Vs))    tau_Vs    <- ep$tau_Vs
    if (is.null(I_slow))    I_slow    <- ep$I_slow
    if (is.null(U_Vs))    U_Vs    <- ep$U_Vs
    if (is.null(max_spike_rate))              max_spike_rate              <- ep$max_spike_rate
    if (is.null(transmission_velocity))            transmission_velocity            <- ep$transmission_velocity
    if (is.null(spine_density))                    spine_density                    <- ep$spine_density
    if (is.null(axon_target))                      axon_target                      <- ep$axon_target
    if (is.null(I_spike))                          I_spike                          <- ep$I_spike
    if (is.null(spike_potential))                  spike_potential                  <- ep$spike_potential
    if (is.null(resting_potential))                resting_potential                <- ep$resting_potential
    if (is.null(threshold))                        threshold                        <- ep$threshold
    if (is.null(leak_conductance))                 leak_conductance                 <- ep$leak_conductance
    if (is.null(axon_branch_count))                axon_branch_count                <- ep$axon_branch_count
    if (is.null(dendrite_branch_count))            dendrite_branch_count            <- ep$dendrite_branch_count
    if (is.null(branch_independence))              branch_independence              <- ep$branch_independence
    if (is.null(branch_spread))                    branch_spread                    <- ep$branch_spread
    if (is.null(apical_target_layer))              apical_target_layer              <- ep$apical_target_layer
    modify_cell_type(type_name, list(
      type_name                        = type_name,
      valence                          = as.integer(valence),
      tau_fast                         = tau_fast, 
      tau_slow                         = tau_slow, 
      tau_Vs                           = tau_Vs, 
      I_slow                           = I_slow, 
      U_Vs                             = U_Vs, 
      max_spike_rate                   = max_spike_rate,
      transmission_velocity            = transmission_velocity,
      spine_density                    = spine_density,
      axon_target                      = axon_target,
      I_spike                          = I_spike,
      spike_potential                  = spike_potential,
      resting_potential                = resting_potential,
      threshold                        = threshold,
      leak_conductance                 = leak_conductance,
      axon_branch_count                = as.integer(axon_branch_count),
      dendrite_branch_count            = as.integer(dendrite_branch_count),
      branch_independence              = branch_independence,
      branch_spread                    = branch_spread,
      apical_target_layer              = apical_target_layer
    ))
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
      layer = "spiny_stellate",
      L1 = "Neurogliaform_cell",
      L2 = "pyramidal",
      L3 = "pyramidal",
      L4 = "spiny_stellate",
      L5 = "pyramidal",
      L6 = "pyramidal_L6"
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
#' @param projection_conductance Numeric giving overall strength of the projection, as synaptic conductance (default: 1e-10, which assumes implicit units of millisiemens).
#' @param presynaptic_type Character string giving type of presynaptic neuron, e.g. "excitatory", "inhibitory", etc. (default: "principal").
#' @param postsynaptic_type Character string giving type of postsynaptic neuron, e.g. "excitatory", "inhibitory", etc. (default: "principal").
#' @param max_col_shift_up Maximum number of columns upwards (increasing columnar indexes) that the projection can reach (default: 0, should be positive integer).
#' @param max_col_shift_down Maximum number of columns downwards (decreasing columnar indexes) that the projection can reach (default: 0, should be positive integer).
#' @param hem_shift Hemisphere shift for the projection: 0 = same hemisphere (default), 1 = contralateral hemisphere. Ignored when the network has only one hemisphere.
#' @return The updated motif object with the new projection loaded.
#' @export
load.projection.into.motif <- function(
    motif,
    presynaptic_layer,
    postsynaptic_layer,
    projection_conductance = 1e-10,
    presynaptic_type = "principal",
    postsynaptic_type = "principal",
    max_col_shift_up = 0,
    max_col_shift_down = 0,
    hem_shift = 0L
  ) {
   
    # Check length of presynaptic_layer
    if (length(presynaptic_layer) != 1) {stop("presynaptic_layer must be a single layer name.")}
    
    # Check length of type inputs
    if (length(presynaptic_type) != 1) {stop("presynaptic_type must be a single type name.")}
    if (length(postsynaptic_type) != 1) {stop("postsynaptic_type must be a single type name.")}
    
    # Get length of postsynaptic_layer input
    n_post_layers <- length(postsynaptic_layer)
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
      proj$pre_type  <- presynaptic_type
      proj$pre_layer <- presynaptic_layer
      proj$post_type <- postsynaptic_type[i]
      proj$post_layer <- postsynaptic_layer[i]
      proj$hem_shift <- as.integer(hem_shift)
      # Add projection to motif
      motif$load_projection(
        proj,
        as.integer(max_col_shift_up),
        as.integer(max_col_shift_down),
        projection_conductance
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
#' @param patch_depth Integer giving the number of "patches" (n_layers x n_columns sheets) in the network.
#' @param layer_height Numeric giving height of each layer (default value is 180.0, which assumes an implicit unit of micron).
#' @param column_diameter Numeric giving diameter of each column (default value is 120.0, which assumes an implicit unit of micron).
#' @param segment_length Numeric giving expected length of each segment in the axonal and dendritic processes of each neuron (default value is 20.0, which assumes an implicit unit of micron).
#' @param layer_separation_factor Numeric giving mean distance between layers as a fraction of layer height (default: 2.5).
#' @param column_separation_factor Numeric giving mean distance between columns as a fraction of column diameter (default: 2.5).
#' @param patch_separation_factor Numeric giving mean distance between network patches as a fraction of column diameter (default: 2.5). 
#' @param neurons_per_node Matrix giving mean number of neurons of each type per node in each cortical layer; dimensions must match n_layers (rows) and length of neuron_types (columns).
#' @param local_synaptic_conductance List (one entry per cortical layer) of matrices giving synaptic conductance in millisiemens for local connections by cell-type; each matrix must have dimensions matching length of neuron_types (rows and columns).
#' @param synaptic_neighborhood Numeric giving the radius (in microns) within which an axon node will trigger a synapse when near a dendrite node (default: 10.0).
#' @param n_hemispheres Integer giving number of hemispheres; must be 1 or 2 (default: 1).
#' @param hemisphere_names Character vector of length n_hemispheres giving names for the hemispheres (default: auto-generated as "left" or c("left","right")).
#' @param hem_separation_factor Numeric giving distance between hemispheres as a fraction of column diameter (default: 5.0).
#' @param n_subcortical_layers Integer giving number of subcortical layers (e.g., thalamic relay nuclei); can be 0 (default: 0).
#' @param subcortical_layer_names Character vector of length n_subcortical_layers giving names for the subcortical layers (default: auto-generated as "subL1", "subL2", ...). Must be distinct from all cortical layer names.
#' @param sub_separation_factor Numeric giving distance from the cortical sheet to the first subcortical layer as a fraction of layer height (default: 5.0).
#' @param subcortical_neurons_per_node Matrix giving mean number of neurons of each type per node in each subcortical layer; dimensions must match n_subcortical_layers (rows) and length of neuron_types (columns). Required if n_subcortical_layers > 0.
#' @param subcortical_local_conductance List (one entry per subcortical layer) of matrices giving synaptic conductance for local connections in each subcortical layer. Required if n_subcortical_layers > 0; can be a single matrix or scalar (broadcast to all layers).
#' @return The updated network object with the specified structure and local nodes generated.
#' @export
set.network.structure <- function(
    network,
    neuron_types = c("principal"),
    layer_names = c("layer"),
    n_layers = 1,
    n_columns = 1,
    patch_depth = 1,
    layer_height = 180.0,
    column_diameter = 120.0,
    segment_length = 20.0, 
    layer_separation_factor = 2.5,
    column_separation_factor = 2.5,
    patch_separation_factor = 2.5,
    neurons_per_node = 10,
    local_synaptic_conductance = 1e-10,
    synaptic_neighborhood = 10.0,
    n_hemispheres = 1,
    hemisphere_names = NULL,
    hem_separation_factor = 5.0,
    n_subcortical_layers = 0,
    subcortical_layer_names = NULL,
    sub_separation_factor = 5.0,
    subcortical_neurons_per_node = NULL,
    subcortical_local_conductance = list()
  ) {
    
    # Check hemisphere count and auto-generate hemisphere names
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
      subcortical_layer_names <- character(0)
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
   
    # Unpack "principal" neuron types 
    if ("principal" %in% neuron_types) {
      
      # ... remake neuron_types
      principals_by_layer <- sapply(layer_names, function(ln) principal.neurons()[[ln]])
      principals <- unique(principals_by_layer)
      principal_idx <- which(neuron_types == "principal")
      neuron_types <- c(principals, neuron_types[-principal_idx])
      n_p <- length(principals)
      n_t <- length(neuron_types)
      nn_p_range <- c(min(n_p + 1, n_t):n_t)
      n_types_old <- n_t - n_p + 1
      
      # ... remake neurons_per_node
      neurons_per_node_new <- matrix(NA, nrow = n_layers, ncol = n_t)
      if (length(neurons_per_node) >= n_types_old) {
        if (!is.null(dim(neurons_per_node))) { # counts per layer and per type specified
          for (i in c(1:nrow(neurons_per_node))) {
            principal_counts <- c()
            for (t in seq_along(principals)) {
              if (principals[t] == principal.neurons()[[layer_names[i]]]) {
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
      
      # ... remake local_synaptic_conductance
      if (!("list" %in% class(local_synaptic_conductance))) {
        
        # Given a single matrix or numeric value
        if ("matrix" %in% class(local_synaptic_conductance) || "numeric" %in% class(local_synaptic_conductance)) {
          rm <- as.matrix(local_synaptic_conductance)
          rm_new <- matrix(0, nrow = n_t, ncol = n_t)
          local_synaptic_conductance <- list()
          for (l in seq_len(n_layers)) {
            if (length(rm) != n_types_old^2) {
              if (length(rm) == 1) {
                # Remake the matrix
                rm_new[nn_p_range, nn_p_range] <- rm
                for (t in seq_along(principals)) {
                  if (principals[t] == principal.neurons()[[layer_names[l]]]) {
                    rm_new[t, t] <- rm
                    rm_new[t, nn_p_range] <- rm
                    rm_new[nn_p_range, t] <- rm
                  }
                }
              } else {
                stop("Dimensions of local_synaptic_conductance matrix must match length of neuron_types, or be a single numeric scalar.")
              }
            } else {
              # Remake the matrix
              if (length(rm[-principal_idx, -principal_idx]) > 0) {
                rm_new[nn_p_range, nn_p_range] <- rm[-principal_idx, -principal_idx]
                for (t in seq_along(principals)) {
                  if (principals[t] == principal.neurons()[[layer_names[l]]]) {
                    rm_new[t, t] <- rm[principal_idx, principal_idx]
                    rm_new[t, nn_p_range] <- rm[principal_idx, -principal_idx]
                    rm_new[nn_p_range, t] <- rm[-principal_idx, principal_idx]
                  }
                }
              }
            }
            local_synaptic_conductance[[l]] <- rm_new
          }
          
        } else {
          stop("local_synaptic_conductance must be a list of matrices, a single matrix, or a single numeric scalar.")
        }
        
      } else { 
        
        # Given a list (... hopefully of matrices)
        for (l in seq_along(local_synaptic_conductance)) {
          rm <- local_synaptic_conductance[[l]]
          rm_new <- matrix(0, nrow = n_t, ncol = n_t)
          # Check if we have a matrix
          if (length(dim(rm)) != 2) {
            stop(paste0("local_synaptic_conductance[[", l, "]] must be a matrix."))
          }
          # Check dimensions 
          if (ncol(rm) != n_types_old) {
            if (ncol(rm) != nrow(rm)) {
              stop(paste0("Dimensions of local_synaptic_conductance[[", l, "]] must match length of neuron_types."))
            }
          }
          # Set new recurrence matrix 
          rm_new[nn_p_range, nn_p_range] <- rm[-principal_idx, -principal_idx]
          for (t in seq_along(principals)) {
            if (principals[t] == principal.neurons()[[layer_names[l]]]) {
              rm_new[t, t] <- rm[principal_idx, principal_idx]
              if (ncol(rm[principal_idx, -principal_idx]) > 0) rm_new[t, nn_p_range] <- rm[principal_idx, -principal_idx]
              if (nrow(rm[-principal_idx, principal_idx]) > 0) rm_new[nn_p_range, t] <- rm[-principal_idx, principal_idx]
            }
          }
          local_synaptic_conductance[[l]] <- rm_new
        }
        
      }
      
    }
   
    # Grab number of neuron types
    n_neuron_types <- length(neuron_types)
    
    # Check neuron counts per node
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
    
    # Helper: coerce conductance input to a list of n x n matrices
    coerce_conductance <- function(cond, n_lyrs, n_types, label) {
      if (!("list" %in% class(cond))) {
        if ("matrix" %in% class(cond) || "numeric" %in% class(cond)) {
          rm <- as.matrix(cond)
          if (length(rm) == 1) {
            rm <- matrix(rm, nrow = n_types, ncol = n_types)
          } else if (length(rm) != n_types^2) {
            stop(paste0("Dimensions of ", label, " matrix must match length of neuron_types, or be a single numeric scalar."))
          }
          out <- list()
          for (l in seq_len(n_lyrs)) out[[l]] <- rm
          return(out)
        } else {
          stop(paste0(label, " must be a list of matrices, a single matrix, or a single numeric scalar."))
        }
      } else if (length(cond) != n_lyrs) {
        stop(paste0("Length of ", label, " list must match ", n_lyrs, "."))
      } else {
        for (l in seq_len(n_lyrs)) {
          sc_dim <- dim(cond[[l]])
          if (length(sc_dim) != 2) stop(paste0(label, "[[", l, "]] must be a matrix."))
          if (any(sc_dim != c(n_types, n_types))) stop(paste0("Dimensions of ", label, "[[", l, "]] must match length of neuron_types."))
        }
        return(cond)
      }
    }
    
    # Check / coerce cortical conductance values
    local_synaptic_conductance <- coerce_conductance(
      local_synaptic_conductance, n_layers, n_neuron_types, "local_synaptic_conductance"
    )
    
    # Check / coerce subcortical conductance values
    if (n_subcortical_layers > 0) {
      if (length(subcortical_local_conductance) == 0) {
        stop("subcortical_local_conductance is required when n_subcortical_layers > 0.")
      }
      subcortical_local_conductance <- coerce_conductance(
        subcortical_local_conductance, n_subcortical_layers, n_neuron_types, "subcortical_local_conductance"
      )
    }
    
    # Build combined neurons_per_node matrix (cortical rows first, then subcortical)
    if (n_subcortical_layers > 0) {
      if (is.null(subcortical_neurons_per_node)) {
        stop("subcortical_neurons_per_node is required when n_subcortical_layers > 0.")
      }
      # Coerce to matrix
      if (is.null(dim(subcortical_neurons_per_node))) {
        if (length(subcortical_neurons_per_node) == 1) {
          subcortical_neurons_per_node <- matrix(subcortical_neurons_per_node, nrow = n_subcortical_layers, ncol = n_neuron_types)
        } else if (length(subcortical_neurons_per_node) == n_neuron_types) {
          subcortical_neurons_per_node <- matrix(rep(subcortical_neurons_per_node, n_subcortical_layers), nrow = n_subcortical_layers, ncol = n_neuron_types, byrow = TRUE)
        } else {
          stop("subcortical_neurons_per_node must have n_subcortical_layers rows and length(neuron_types) columns.")
        }
      }
      if (any(dim(subcortical_neurons_per_node) != c(n_subcortical_layers, n_neuron_types))) {
        stop("subcortical_neurons_per_node must have n_subcortical_layers rows and length(neuron_types) columns.")
      }
      nrn_per_node_combined <- rbind(neurons_per_node, subcortical_neurons_per_node)
    } else {
      nrn_per_node_combined <- neurons_per_node
    }
    
    # Set structure (new C++ signature: hsl_names as list, n as 5-vector, sep_factor as 5-vector)
    network$set_network_structure(
      neuron_types,
      list(hemisphere_names, subcortical_layer_names, layer_names),  # hsl_names: {hem, sub, lyr}
      as.integer(c(n_hemispheres, n_subcortical_layers, n_layers, n_columns, patch_depth)),  # n
      c(hem_separation_factor, sub_separation_factor, layer_separation_factor, column_separation_factor, patch_separation_factor),  # sep_factors
      layer_height,
      column_diameter,
      segment_length, 
      synaptic_neighborhood,
      nrn_per_node_combined,
      local_synaptic_conductance,
      subcortical_local_conductance
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
#' @param verbose Logical indicating whether to print a summary of the fetched components (default: TRUE).
#' @return A list containing the components of the network.
#' @export
fetch.network.components <- function(
    network, 
    include_arbors = FALSE, 
    verbose = TRUE
  ) {
    
    # Grab raw components
    network.components <- network$fetch_network_components(include_arbors)
    
    # Compute synapse distribution
    if (include_arbors) {
      n_neurons <- network.components$n_neurons
      arbors <- network.components$arbors
      synapse_info <- as.data.frame(matrix(NA, nrow = n_neurons, ncol = 3))
      colnames(synapse_info) <- c("neuron_idx", "neuron_type", "n_synapses") # n_synapses is the number of times this neuron synapses onto another cell
      synapse_info$neuron_idx <- c(1:n_neurons)
      for (n in c(1:n_neurons)) {
        synapse_info[n, "neuron_type"] <- network.components$neuron_type_name[n]
        mask <- arbors[, "neuron_idx"] == n
        synapse_info[n, "n_synapses"] <- sum(arbors[mask, "is_synapse"])
      }
      network.components$synapse_info <- synapse_info
    }
    
    # Print summary
    if (verbose) {
      cat("Summary of network:\n")
      cat("\tNumber of neurons:", network.components$n_neurons, "\n")
      if (include_arbors) {
        cat("\tNumber of synapses:", sum(network.components$arbors[,"is_synapse"]), "\n")
      }
      cat("\tHemisphere names:", paste(network.components$hem_names, collapse = ", "), "\n")
      cat("\tNumber of hemispheres:", network.components$n_hem, "\n")
      cat("\tSubortical layer names:", paste(network.components$sub_names, collapse = ", "), "\n")
      cat("\tNumber of subcortical layers:", network.components$n_sub, "\n")
      cat("\tCortical layer names:", paste(network.components$layer_names, collapse = ", "), "\n")
      cat("\tNumber of cortical layers:", network.components$n_layers, "\n")
      cat("\tNumber of columns:", network.components$n_columns, "\n")
      cat("\tNumber of patches:", network.components$n_patches, "\n")
      cat("\tCell types used:", paste(unique(network.components$neuron_type_name), collapse = ", "), "\n")
      if (network.components$n_neurons > 0) {
        cat("\tMotifs used:", paste(network.components$edge_type_names, collapse = ", "), "\n")
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
#' @param verbose Logical indicating whether to print progress messages during motif application (default: TRUE).
#' @return The updated network object with the motif applied.
#' @export
apply.circuit.motif <- function(
    network,
    motif,
    verbose = FALSE
  ) {
    network$apply_circuit_motif(motif, verbose)
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
#'  soma_size_factor = 3.0, 
#'  return_plot = FALSE,
#'  return_cell_arbor_idx = FALSE
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
#' @param soma_size_factor Numeric value controlling how cell size in the plot scales to the number of cells. 
#' @param return_plot Logical indicating whether to return the ggplot object or print the plot directly (default: TRUE).
#' @param return_cell_arbor_idx Logical indicating whether to return the soma_mask and arbor_idx used for plotting or not (default: TRUE).
#' @param units_distance Character string giving the units of distance, value only used to label the plot (Default: "micron").
#' @return Either prints the plot directly or returns the ggplot object, depending on the value of return_plot.
#' @export
plot.network <- function(
    network,
    soma_mask = NULL,
    arbor_idx = NULL,
    threedim = FALSE,
    title = NULL,
    soma_density = 1.0,
    arbor_density = 0.01,
    arbor_cell_type = "all",
    plot_motif = "all",
    reconstruct_arbors = TRUE,
    edge_color = "pre_type",
    soma_color = "layer",
    soma_size_factor = 1.0,
    return_plot = TRUE,
    return_cell_arbor_idx = TRUE,
    units_distance = "microns"
  ) {
    
    # Get network components
    ntw <- network$fetch_network_components(reconstruct_arbors) # Retrieve arbors? 
    
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
    
    if (is.null(soma_mask)) {
      # Get number of cell bodies to plot 
      n_soma <- 1
      if (soma_density > 0) {
        n_soma <- round(ntw$n_neurons * min(1, soma_density))
      }
      
      # Make mask for soma
      soma_idx <- sort(sample(ntw$n_neurons, n_soma, replace = FALSE))
      soma_mask <- rep(FALSE, ntw$n_neurons)
      soma_mask[soma_idx] <- TRUE
    } else {
      n_soma <- sum(soma_mask)
    }
    
    # Get cell coordinates and types 
    neuron_coordinates <- ntw$coordinates_spatial[soma_mask,]
    neuron_types <- ntw$neuron_type_name[soma_mask]
    
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
    neuron_layer <- as.factor(neuron_layer_labels)
    
    # Find range of possible neurons for arbor plotting, based on cell type if specified
    edge_celltype_mask <- TRUE
    if (arbor_cell_type != "all") {
      edge_celltype_mask <- FALSE 
      for (act in arbor_cell_type) {
        edge_celltype_mask <- edge_celltype_mask | neuron_types == act
      }
      if (sum(edge_celltype_mask) == 0) {
        stop("No neurons match the specified arbor_cell_type.")
      }
    }
    
    # Randomly select neurons for arbor plotting from among those that are in the soma plot and match the specified cell type (if applicable)
    if (is.null(arbor_idx)) {
      arbor_mask <- soma_mask & edge_celltype_mask
      soma_idx_ct <- which(arbor_mask)
      n_arbors <- 1
      if (arbor_density > 0) {
        n_arbors <- round(sum(arbor_mask) * min(1, arbor_density)) 
      }
      arbor_idx <- sort(sample(soma_idx_ct, n_arbors, replace = FALSE))
    } 
    
    # Create cells dataframe
    y_coord <- "y"
    z_coord <- "z"
    if (threedim) {
      y_coord <- "z"
      z_coord <- "y"
    }
    soma <- data.frame(
      idx = c(1:n_soma), 
      x = neuron_coordinates[,"x"], 
      y = neuron_coordinates[,y_coord],
      z = neuron_coordinates[,z_coord],
      layer = neuron_layer,
      type = neuron_types
    )
    
    # Get cell edge pairs / reconstruct arbors
    synapses_included <- FALSE
    if (reconstruct_arbors) {
      
      # Reconstruct the edge matrix for these cells only
      edges <- ntw[["arbors"]]
      
      # Find the number of edges in reconstruction
      n_downsample_edges <- c()
      for (n in arbor_idx) {
        n_edges <- sum(edges[,"neuron_idx"] == n)
        n_downsample_edges <- c(n_downsample_edges, n_edges)
      }
      
      # Make downsampled matrix
      edges_downsampled <- matrix(NA, nrow = sum(n_downsample_edges), ncol = ncol(edges))
      idx_start <- 1
      idx_end <- 0
      for (i in seq_along(arbor_idx)) {
        idx_start <- idx_end + 1
        idx_end <- idx_end + n_downsample_edges[i]
        edges_downsampled[idx_start:idx_end, ] <- edges[edges[,"neuron_idx"] == arbor_idx[i], ]
      }
      
      # Add colnames: "neuron_idx", "arbor_id", "is_axon", "node_type", "parent_idx", "is_leaf", "is_synapse", "z_start", "y_start", "x_start", "z_end", "y_end", "x_end"
      # ... note: "neuron_idx" is for the pre-synaptic cell, i.e. = "pre_idx" below. 
      colnames(edges_downsampled) <- colnames(edges)
      
      # Make into data frame and rename axons and node type
      edges_downsampled <- as.data.frame(edges_downsampled) 
      edges_downsampled$is_axon[edges_downsampled$is_axon == 1] <- "axon"
      edges_downsampled$is_axon[edges_downsampled$is_axon == 0] <- "dendrite"
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
      edges_downsampled$pre_type <- neuron_types[edges_downsampled$neuron_idx]
      
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
      et_masked <- seq_along(edge_type_names)
      if (plot_motif != "all") {
        edge_type_mask <- edge_type_names %in% plot_motif
        if (sum(edge_type_mask) == 0) {
          stop("No edge types match the specified plot_motif.")
        }
        edge_type_names <- edge_type_names[edge_type_mask]
        et_masked <- which(edge_type_mask)
      }
      
      # Collect edges by motifs
      edges <- matrix(0, nrow = 0, ncol = 5)
      for (et in seq_along(edge_type_names)) {
        et_name <- edge_type_names[et]
        et_edges <- ntw$edge_idx_by_type[[et_masked[et]]]
        for (ni in unique(et_edges[, "pre_neuron_idx"])) {
          if (sum(ni == arbor_idx) == 0) {
            ni_mask <- et_edges[, "pre_neuron_idx"] != ni
            et_edges <- et_edges[ni_mask, ]
          }
        }
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
      
      # Find coordinates for start and end of edges
      edges$x_start <- soma[edges$pre_idx, "x"]
      edges$y_start <- soma[edges$pre_idx, "y"]
      edges$z_start <- soma[edges$pre_idx, "z"]
      edges$x_end <- soma[edges$post_idx, "x"]
      edges$y_end <- soma[edges$post_idx, "y"]
      edges$z_end <- soma[edges$post_idx, "z"]
      
    }
    
    # Set point size to scale with number of cells
    soma_size <- soma_size_factor * 10 / log(n_soma + 1)
    
    # Make colors 
    if (length(unique(as.character(soma[,soma_color]))) == 1) soma[,soma_color] <- "cell"
    colored_labels <- unique(
      c(unique(as.character(edges[,edge_color])), 
        unique(as.character(soma[,soma_color])))
      )
    known_label_colors <- list(
      "cell" = "gray50",
      "layer" = "gray50",
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
      "pyramidal" = "green4",
      "pyramidal_L6" = "green4",
      "spiny_stellate" = "green2",
      "interneuron" = "red",
      "inhibitory" = "red", 
      "Neurogliaform_cell" = "red", 
      "PV" = "darkred",
      "SOM" = "darkorchid",
      "SST" = "darkorchid",
      "VIP" = "darkorange",
      "axon" = "green3",
      "dendrite" = "darkred"
    )
    unknown_label_colors <- c("aquamarine1", "gray95", "gray55", "gray75", "cyan", "cornflowerblue", "coral", "burlywood", "darkolivegreen")
    label_colors <- rep("white", length(colored_labels))
    names(label_colors) <- colored_labels
    for (cl in seq_along(colored_labels)) {
      label <- colored_labels[cl]
      hit_mask <- label == names(known_label_colors)
      if (any(hit_mask)) {
        hit_idx <- which(hit_mask)[1]
        label_colors[cl] <- known_label_colors[[hit_idx]]
      } else {
        label_colors[cl] <- sample(unknown_label_colors, 1)
      }
    }
    
    # Add color for synapses
    syn_color <- "orange"
    if (synapses_included) {
      label_colors <- c(label_colors, syn_color)
      names(label_colors)[length(label_colors)] <- "syn"
    }
    
    # Plot
    title_size <- 14 
    axis_size <- 12 
    legend_size <- 10
    
    if (threedim) {
      
      # Translate colors for plotly
      hex <- rgb(t(col2rgb(label_colors)), maxColorValue = 255)
      
      # Remake level names for plotly
      if (edge_color == "is_axon") {
        level_names <- c("axon", "dendrite", ntw$layer_names)
      } else if (edge_color == "pre_type") {
        level_names <- c(unique(edges$pre_type), ntw$layer_names)
      } else {
        stop("edge_color must be 'is_axon' or 'pre_type' when reconstructing arbors.")
      }
      if (synapses_included) {
        level_names <- c(level_names, "syn")
        # ... and reset levels in synapse_coordinates
        synapse_coordinates$syn <- "syn" 
        synapse_coordinates$syn <- factor(synapse_coordinates$syn, levels = level_names, labels = level_names)
      }
      
      # Reset levels in soma and edge data frames
      soma$layer <- factor(soma$layer, levels = level_names, labels = level_names)
      edges[,edge_color] <- factor(edges[,edge_color], levels = level_names, labels = level_names) 
      
      # Make long version of edges for faster ploting in plotly
      edges_long <- data.frame(
        x = c(rbind(edges$x_start, edges$x_end, NA)),
        y = c(rbind(edges$y_start, edges$y_end, NA)),
        z = c(rbind(edges$z_start, edges$z_end, NA)),
        group = rep(edges[[edge_color]], each = 3)
      )
      
      # Initialize plotly plot with edges
      plt <- plotly::plot_ly(
        edges_long,
        x = ~x,
        y = ~z,
        z = ~y,
        type = "scatter3d",
        mode = "lines",
        color = ~factor(group),
        colors = hex
      )
      
      # Add soma as points
      plt <- plt |>
        plotly::add_trace(
          data = soma,
          x = ~x,
          y = ~y,
          z = ~z,
          type = "scatter3d",
          mode = "markers",
          marker = list(size = soma_size),
          color = ~factor(layer),
          colors = hex
        ) 
      
      # Add synapses
      if (synapses_included) {
        plt <- plt |> 
          plotly::add_trace(
            data = synapse_coordinates,
            x = ~x,
            y = ~z,
            z = ~y,
            type = "scatter3d",
            mode = "markers",
            marker = list(size = soma_size/2),
            color = ~syn,
            colors = hex
          )
      }
      
      # Label axes and fix colors into light mode 
      plt <- plt |>
        plotly::layout(
          template = "plotly_white",
          paper_bgcolor = "white",
          plot_bgcolor = "white",
          font = list(color = "black"),
          scene = list(
            xaxis = list(title = "Cortical Columns", color = "black", backgroundcolor = "white"),
            zaxis = list(title = "Cortical Layers", color = "black", backgroundcolor = "white"),
            yaxis = list(title = "Cortical Patches", color = "black", backgroundcolor = "white")
          )
        )
      
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
          x = paste0("columnar coordinate (", units_distance, ")"), 
          y = paste0("laminar coordinate (", units_distance, ")")
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
      
      # Add synapses 
      if (synapses_included) {
        plt <- plt + 
          ggplot2::geom_point(
            data = synapse_coordinates,
            ggplot2::aes(x = x, y = y, color = "syn"),
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
#' @usage plot.network.traces(network, return_plot)
#' @param network Network object with SGT simulation traces to plot.
#' @param return_plot Logical indicating whether to return the ggplot object (TRUE) or print it (FALSE) (default: FALSE).
#' @param input_matrix Matrix of stimulus currents, with rows representing neurons and columns representing sample times. Presumably the one used to generate the traces. Options. If provided, will be added to the bottom of the plot. 
#' @return A ggplot object showing spike traces for all neurons in the network over time.
#' @export
plot.network.traces <- function(
    network,
    return_plot  = FALSE, 
    input_matrix = NULL
  ) {
    
    # Get the traces to print
    v_traces <- network$fetch_sim_results()$v_traces
    
    # Get network components
    ntw <- network$fetch_network_components(FALSE) # Retrieve arbors?
    
    # Initialize R data frame for ggplot
    v_traces_long <- data.frame()
    time_seq        <- seq(1, by = ntw$sim_dt, length.out = ncol(v_traces))
    sim_steps       <- c(1:ncol(v_traces))
    for (i in 1:nrow(v_traces)) {
      neuron_trace <- data.frame(
        time      = time_seq,
        potential = v_traces[i, sim_steps],
        id        = i,
        type      = ntw$neuron_type_name[i]
      )
      v_traces_long <- rbind(v_traces_long, neuron_trace)
    }
    v_traces_long$id <- as.character(v_traces_long$id)
    
    # Make plot
    title_size  <- 14 
    axis_size   <- 12 
    legend_size <- 10
    plt <- ggplot2::ggplot(v_traces_long, ggplot2::aes(x = time, y = potential, group = id, color=id)) +
      ggplot2::geom_line() +
      ggplot2::facet_wrap(~ type, ncol = 1) +
      ggplot2::theme_minimal() +
      ggplot2::theme(
        panel.background = ggplot2::element_rect(fill = "white", colour = NA),
        plot.background  = ggplot2::element_rect(fill = "white", colour = NA),
        plot.title       = ggplot2::element_text(hjust = 0.5, size = title_size),
        axis.title       = ggplot2::element_text(size = axis_size),
        axis.text        = ggplot2::element_text(size = axis_size),
        legend.title     = ggplot2::element_text(size = legend_size),
        legend.text      = ggplot2::element_text(size = legend_size),
        legend.position  = "none") +
      ggplot2::labs(
        title = "SGT Simulation Traces",
        x     = paste0("Time (ms)"),
        y     = paste0("Membrane Potential (mV)")
      )
    
    # Add input matrix 
    if (!is.null(input_matrix)) {
      
      # Remove x axis
      plt <- plt +
        ggplot2::theme(
          axis.title.x = ggplot2::element_blank(),
          axis.ticks.x = ggplot2::element_blank(),
          axis.text.x  = ggplot2::element_blank()
        )
      
      # Make stimulus plot
      df <- data.frame(
        t = rep(seq_len(ncol(input_matrix)) * ntw$sim_dt, each = nrow(input_matrix)), 
        x = as.numeric(input_matrix)
      )
      
      plt_stim <- ggplot2::ggplot(df, ggplot2::aes(t, x)) +
        ggplot2::geom_line(linewidth = 0.6) +
        ggplot2::theme_minimal() +
        ggplot2::xlab("Time (ms)") +
        ggplot2::ylab("Stimulus (pA)")
      
      # Stack them
      plt <- patchwork::wrap_plots(plt, plt_stim, ncol = 1, heights = c(4, 1))
      
    }
    
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
#' @param dt Time step length in the implicit time units of the network (default: 1e-3, which is 1 micosecond time steps, assuming an implicit time unit of ms).
#' @param initial_potential Initial value for membrane potential, applied to all cells (Default is -70 mV).
#' @return List containing the following elements: \item{v_traces}{Matrix of simulated spike traces for all neurons over time (neurons as rows, sample times as columns).} \item{spike_counts}{Vector of spike counts for each neuron in the network.} 
#' @export
run.SGT <- function(
    network,
    stimulus_current_matrix, 
    dt = 1e-3,  
    initial_potential = -70.0
  ) {
    network$SGT(stimulus_current_matrix, dt, initial_potential)
    return(network$fetch_sim_results())
  }
