
# Clear the R workspace to start fresh
rm(list = ls())

# Set seed for reproducibility
set.seed(12345) 

# Load neurons package
library(DACx, quietly = TRUE) 

cortical.patch <- new.network()
cortical.patch <- set.network.structure(
  cortical.patch,
  neuron_types = c("principal", "PV", "SST"),
  layer_names = c("L6", "L5", "L4", "L3", "L2", "L1"),
  n_columns = 8,
  patch_depth = 4,
  layer_separation_factor = 3.0,
  column_separation_factor = 4.5,
  patch_separation_factor = 5.5,
  neurons_per_node = matrix(c(
    10, 5, 5, # L6
    10, 5, 5, # L5
    10, 5, 5, # L4
    10, 5, 5, # L3
    10, 5, 5, # L2
    5, 0, 1   # L1
    ), ncol = 3, byrow = TRUE)
)

# Add mesoscale motifs
motif.pp <- new.motif(motif_name = "principal projections")
motif.pp <- load.projection.into.motif(motif.pp, "L4", "L2", 0.9)
motif.pp <- load.projection.into.motif(motif.pp, "L4", "L3", 0.9)
motif.pp <- load.projection.into.motif(motif.pp, "L4", "L5", 0.25)
motif.pp <- load.projection.into.motif(motif.pp, "L4", "L6", 0.25)
motif.pp <- load.projection.into.motif(motif.pp, "L2", "L5")
motif.pp <- load.projection.into.motif(motif.pp, "L3", "L5")
motif.pp <- load.projection.into.motif(motif.pp, "L5", "L2")
motif.pp <- load.projection.into.motif(motif.pp, "L5", "L3")
motif.pp <- load.projection.into.motif(motif.pp, "L5", "L6", 0.25)
motif.pp <- load.projection.into.motif(motif.pp, "L6", "L4", 0.25)
cortical.patch <- apply.circuit.motif(
  cortical.patch,
  motif.pp
)

motif.ACxlat <- new.motif(motif_name = "ACx laterals")
# Add projection for each layer
for (layer in c("L1", "L2", "L3", "L4", "L5", "L6")) {
  # Excitatory laterals 
  for (celltype in c("principal", "PV", "SST")) {
    motif.ACxlat <- load.projection.into.motif(
      motif.ACxlat, 
      presynaptic_layer = layer, 
      postsynaptic_layer = layer, 
      presynaptic_type = "principal", 
      postsynaptic_type = celltype,
      max_col_shift_up = 4,
      max_col_shift_down = 4
    )
  }
  # Inhibitory laterals
  motif.ACxlat <- load.projection.into.motif(
    motif.ACxlat, 
    presynaptic_layer = layer, 
    postsynaptic_layer = layer, 
    presynaptic_type = "SST", 
    postsynaptic_type = "principal",
    max_col_shift_up = 4,
    max_col_shift_down = 4
  )
}
cortical.patch <- apply.circuit.motif(
  cortical.patch,
  motif.ACxlat
)
# ... why does this one take so much longer? The "max_col_shift_up" / "max_col_shift_down" = 8 is the issue / difference, compared to the other two motifs, which have the default = 0. 

motif.L6inhib <- new.motif(motif_name = "L6 inhibition")
# Layer 5 projections
motif.L6inhib <- load.projection.into.motif(
  motif.L6inhib, 
  presynaptic_layer = "L5", 
  postsynaptic_layer = c("L4", "L3", "L2"), 
  presynaptic_type = "PV", 
  postsynaptic_type = "principal"
)
# Layer 6 projections
motif.L6inhib <- load.projection.into.motif(
  motif.L6inhib, 
  presynaptic_layer = "L6", 
  postsynaptic_layer = c("L5", "L4", "L3", "L2"), 
  presynaptic_type = "PV", 
  postsynaptic_type = "principal"
)
cortical.patch <- apply.circuit.motif(
  cortical.patch,
  motif.L6inhib
)








##################################################################
# Make plot

ntw <- cortical.patch$fetch_network_components(TRUE)
neuron_coordinates <- ntw$coordinates_spatial
neuron_types <- ntw$neuron_type_name
n_neurons <- ntw$n_neurons
layer_names <- ntw$layer_names
neuron_layer <- as.factor(layer_names[ntw$coordinates_node[,"layer_idx"]])

# Get cell edge pairs

edges <- ntw[["arbors"]]
n_neuron_downsample <- 100
downsample_celltype <- "pyramidal"
type_idx <- which(neuron_types == downsample_celltype)
neuron_downsample_idx <- sample(type_idx, n_neuron_downsample, replace = FALSE)

n_downsample_edges <- c()
for (n in neuron_downsample_idx) {
  n_edges <- sum(edges[,"neuron_idx"] == n)
  n_downsample_edges <- c(n_downsample_edges, n_edges)
}
edges_downsampled <- matrix(NA, nrow = sum(n_downsample_edges), ncol = ncol(edges))
idx_start <- 1
idx_end <- 0
for (i in seq_along(neuron_downsample_idx)) {
  idx_start <- idx_end + 1
  idx_end <- idx_end + n_downsample_edges[i]
  n <- neuron_downsample_idx[i]
  edges_downsampled[idx_start:idx_end, ] <- edges[edges[,"neuron_idx"] == n, ]
}
colnames(edges_downsampled) <- colnames(edges)

edges_downsampled <- as.data.frame(edges_downsampled) 
edges_downsampled$is_axon[edges_downsampled$is_axon == 1] <- "axon"
edges_downsampled$is_axon[edges_downsampled$is_axon == 0] <- "dendrite"

edges_downsampled$seg_length <- sqrt(
  (edges_downsampled[,"x_end"] - edges_downsampled[,"x_start"])^2 +
    (edges_downsampled[,"y_end"] - edges_downsampled[,"y_start"])^2 +
    (edges_downsampled[,"z_end"] - edges_downsampled[,"z_start"])^2
)
hist(edges_downsampled$seg_length)

synapse_coordinates <- edges_downsampled[edges_downsampled$is_synapse > 0, c("z_end", "y_end", "x_end")]
colnames(synapse_coordinates) <- c("z", "y", "x")



# Create cells dataframe
cells <- data.frame(
  idx = c(1:nrow(neuron_coordinates)), 
  x = neuron_coordinates[,"x"],
  y = neuron_coordinates[,"z"],
  z = neuron_coordinates[,"y"],
  layer = neuron_layer,
  type = neuron_types
)

# Set point size to scale with number of cells
cell_size_factor <- 3.0
n_cells <- nrow(cells)
cell_size <- cell_size_factor * 10 / log(n_cells + 1)

# Set arrow size to scale with number of edges_downsampled
n_edges <- nrow(edges_downsampled)

# Scake alpha by number of edges 
edge_alpha <- max(0.1, min(1, n_cells / (n_edges + 1)))

# Make colors 
edge_color <- "is_axon"
cell_color <- "layer"
colored_labels <- unique(
  c(unique(as.character(edges_downsampled[,edge_color])), 
    unique(as.character(cells[,cell_color])))
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
  hit_mask <- grepl(label, names(known_label_colors))
  if (any(hit_mask)) {
    hit_idx <- which(hit_mask)[1]
    label_colors[cl] <- known_label_colors[[hit_idx]]
  } else {
    label_colors[cl] <- sample(unknown_label_colors, 1)
  }
}
label_colors <- c(label_colors, "orange")

library(plotly)

hex <- rgb(t(col2rgb(label_colors)), maxColorValue = 255)

cells$layer <- factor(
  cells$layer,
  levels = c("axon", "dendrite", "L6", "L5", "L4", "L3", "L2", "L1", "syn"),
  labels = c("axon", "dendrite", "L6", "L5", "L4", "L3", "L2", "L1", "syn")
)
edges_downsampled[,edge_color] <- factor(
  edges_downsampled[,edge_color],
  levels = c("axon", "dendrite", "L6", "L5", "L4", "L3", "L2", "L1", "syn"),
  labels = c("axon", "dendrite", "L6", "L5", "L4", "L3", "L2", "L1", "syn")
) 
synapse_coordinates$syn <- "syn" 
synapse_coordinates$syn <- factor(
  synapse_coordinates$syn,
  levels = c("axon", "dendrite", "L6", "L5", "L4", "L3", "L2", "L1", "syn"),
  labels = c("axon", "dendrite", "L6", "L5", "L4", "L3", "L2", "L1", "syn")
)

edges_downsampled_long <- data.frame(
  x = c(rbind(edges_downsampled$x_start, edges_downsampled$x_end, NA)),
  y = c(rbind(edges_downsampled$y_start, edges_downsampled$y_end, NA)),
  z = c(rbind(edges_downsampled$z_start, edges_downsampled$z_end, NA)),
  group = rep(edges_downsampled[[edge_color]], each = 3)
)

plt <- plot_ly(
  edges_downsampled_long,
  x = ~x,
  y = ~z,
  z = ~y,
  type = "scatter3d",
  mode = "lines",
  color = ~factor(group),
  colors = hex
)

plt <- plt |>
  add_trace(
    data = cells,
    x = ~x,
    y = ~y,
    z = ~z,
    type = "scatter3d",
    mode = "markers",
    marker = list(size = cell_size),
    color = ~factor(layer),
    colors = hex
  ) 

plt <- plt |> 
  add_trace(
    data = synapse_coordinates,
    x = ~x,
    y = ~z,
    z = ~y,
    type = "scatter3d",
    mode = "markers",
    marker = list(size = cell_size/2),
    color = ~syn,
    colors = hex
  )

plt <- plt |>
  layout(
    scene = list(
      xaxis = list(title = "Cortical Columns"),
      zaxis = list(title = "Cortical Layers"),
      yaxis = list(title = "Cortical Patches")
    )
  )

plt













# Scale, microns over pixels
sc <- 500/1220

# sizes, pixels
cell_diameter <- 34 
lam <- 2700
col_to_lam <- 1470/1220
col <- col_to_lam * lam
coluster_radius <- 150
n_cols <- col / (2.5 * coluster_radius)

# Convert to microns
cell_diameter <- cell_diameter * sc
lam <- lam * sc # height of patch (laminar axis)
col <- col * sc # width of patch (columnar axis)
coluster_radius <- coluster_radius * sc

cluster_density <- 0.4
cells_per_cluster <- (pi * (coluster_radius)^2) * cluster_density / (pi * (cell_diameter/2)^2) 
total_cells <- cells_per_cluster * n_cols * 5 # five layers

# print results 
cat("cell_diameter (microns) =", cell_diameter,
    "\nlam (microns) =", lam,
    "\ncol (microns) =", col,
    "\ncoluster_radius (microns) =", coluster_radius,
    "\nn_cols =", n_cols,
    "\ncells_per_cluster =", cells_per_cluster,
    "\ntotal_cells =", total_cells)

# At 30e3 microns/ms, will take 0.05 ms to cross a 1500 micron patch
# By default, run growth-transform sim at 1e-3 ms (1 microsecond) time-step





# Clear the R workspace to start fresh
rm(list = ls())
# Set seed for reproducibility
set.seed(12345) 
# Load neurons package
library(neurons) 

cortical.patch <- new.network()

init_known_celltypes()

cortical.patch <- set.network.structure(
  cortical.patch,
  neuron_types = c("principal", "PV", "SST"),
  neurons_per_node = c(10, 5, 5),
  recurrence_factors = 0.75,
  pruning_threshold_factor = 0.1
)

plot.network(cortical.patch)

cortical.patch.comps <- cortical.patch$fetch_network_components()
n_neurons <- cortical.patch.comps$n_neurons


stim_time_ms <- 50
dt <- 1e-3
n_steps <- stim_time_ms/dt

stim_length_ms <- 20
stim_start_ms <- 10
stim_length <- stim_length_ms / dt
stim_start <- stim_start_ms / dt 
stim_end <- stim_start + stim_length - 1

rest_current <- 0.001e-7 # 0.1 pico amp
principal_mask <- cortical.patch.comps$neuron_type_name == "principal"
stimulus_current_matrix <- matrix(rest_current, nrow = n_neurons, ncol = n_steps)
stimulus_current_matrix[principal_mask, stim_start:stim_end] <- 0.001e-6 # one pico amp

spike_traces <- run.GTsim(
  cortical.patch, 
  stimulus_current_matrix,    # matrix of input currents (rows: neurons, columns: time bins)
  dt = dt                     # time step length, in ms
)

spike_traces_long <- data.frame()
samples <- n_steps
end <- n_steps
#end <- samples*10
start <- end - samples + 1
for (i in 1:nrow(spike_traces)) { # 
  neuron_trace <- data.frame(
    time = seq(start, by=dt, length.out=samples),
    potential = spike_traces[i,c(start:end)],
    id = i,
    type = cortical.patch.comps[["neuron_type_name"]][i]
  )
  spike_traces_long <- rbind(spike_traces_long, neuron_trace)
}
spike_traces_long$id <- as.character(spike_traces_long$id)

ggplot2::ggplot(spike_traces_long, ggplot2::aes(x=time, y=potential, group = id, color=id)) +
  ggplot2::geom_line() +
  ggplot2::geom_vline(xintercept=stim_end * dt, linetype="dashed", color="black") +
  ggplot2::geom_vline(xintercept=stim_start * dt, linetype="dashed", color="black") +
  ggplot2::facet_wrap(~ type, ncol=1) +
  ggplot2::theme_minimal() +
  ggplot2::theme(legend.position="none") +
  ggplot2::labs(
    title = "Neuron Spike Traces",
    x = "Time (ms)",
    y = "Membrane Potential (unit_potential)"
  )

# print all individually
for (neuron_id in unique(spike_traces_long$id)) {
  neuron_trace <- spike_traces_long[spike_traces_long$id == neuron_id, ]
  p <- ggplot2::ggplot(neuron_trace, ggplot2::aes(x=time, y=potential)) +
    ggplot2::geom_line(color="blue") +
    ggplot2::geom_vline(xintercept=stim_end * dt, linetype="dashed", color="black") +
    ggplot2::geom_vline(xintercept=stim_start * dt, linetype="dashed", color="black") +
    ggplot2::theme_minimal() +
    ggplot2::labs(
      title = paste("Neuron Spike Trace - ID:", neuron_id, "Type:", unique(neuron_trace$type)),
      x = "Time (ms)",
      y = "Membrane Potential (unit_potential)"
    )
  print(p)
}



