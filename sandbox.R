
# Clear the R workspace to start fresh
rm(list = ls())

# Set seed for reproducibility
set.seed(12345) 

# Load neurons package
library(neurons, quietly = TRUE) 

cortical.patch <- new.network()
cortical.patch <- set.network.structure(
  cortical.patch,
  neuron_types = c("principal", "PV", "SST"),
  layer_names = c("L6", "L5", "L4", "L2/3"),
  n_columns = 8,
  patch_depth = 4,
  layer_separation_factor = 3.0,
  column_separation_factor = 4.5,
  patch_separation_factor = 5.5,
  neurons_per_node = c(10, 5, 5)
)



# Make plot


ntw <- cortical.patch$fetch_network_components(TRUE)
neuron_coordinates <- ntw$coordinates_spatial
neuron_types <- ntw$neuron_type_name

layer_names <- ntw$layer_names
neuron_layer <- as.factor(layer_names[ntw$coordinates_node[,"layer_idx"]])






# Get cell edge pairs
plot_motif <- "local connections"
cell_size_factor <- 5.0
# edges <- matrix(0, nrow = 0, ncol = 5)
# edge_type_names <- ntw$edge_type_names
# edge_type_mask <- edge_type_names %in% plot_motif
# edge_type_names <- edge_type_names[edge_type_mask]
# n_edge_types <- length(edge_type_names)
# et_masked <- which(edge_type_mask)
# for (et in seq_along(edge_type_names)) {
#   et_name <- edge_type_names[et]
#   et_edges <- ntw$edge_idx_by_type[[et_masked[et]]]
#   et_edges <- cbind(
#     et_edges, 
#     rep(et_name, nrow(et_edges)),
#     neuron_types[et_edges[,"pre_neuron_idx"]],
#     neuron_types[et_edges[,"post_neuron_idx"]]
#   )
#   edges <- rbind(edges, et_edges)
# }
# edges <- as.data.frame(edges)
# colnames(edges) <- c("pre_idx", "post_idx", "motif", "pre_type", "post_type")

edges <- matrix(0, nrow = 0, ncol = 7)
edges <- as.data.frame(edges)
colnames(edges) <- c("is_axon", "z_start", "y_start", "x_start", "z_end", "y_end", "x_end")
for (a in ntw[["arbor_list"]]) {
  
  for (b in unique(a[,"is_axon"])) {
    
    ab <- a[a[,"is_axon"] == b,]
    
    # parent rows
    p <- ab[, "parent_idx"]
    
    # keep only rows that correspond to actual segments
    seg_idx <- p != 0
    
    segments <- cbind(
      is_axon = ab[seg_idx, "is_axon"],
      
      z_start = ab[p[seg_idx], "z"],
      y_start = ab[p[seg_idx], "y"],
      x_start = ab[p[seg_idx], "x"],
      
      z_end   = ab[seg_idx, "z"],
      y_end   = ab[seg_idx, "y"],
      x_end   = ab[seg_idx, "x"]
    )
    
    edges <- rbind(edges, segments)
    
  }
  
  
}
edges$is_axon[edges$is_axon == 1] <- "axon"
edges$is_axon[edges$is_axon == 0] <- "dendrite"

# Create cells dataframe
cells <- data.frame(
  idx = c(1:nrow(neuron_coordinates)), 
  x = neuron_coordinates[,"x"],
  y = neuron_coordinates[,"z"],
  z = neuron_coordinates[,"y"],
  layer = neuron_layer,
  type = neuron_types
)

# Find coordinates for start and end of edges
# edges$x_start <- cells[edges$pre_idx, "x"]
# edges$y_start <- cells[edges$pre_idx, "y"]
# edges$z_start <- cells[edges$pre_idx, "z"]
# edges$x_end <- cells[edges$post_idx, "x"]
# edges$y_end <- cells[edges$post_idx, "y"]
# edges$z_end <- cells[edges$post_idx, "z"]

# Set point size to scale with number of cells
n_cells <- nrow(cells)
cell_size <- cell_size_factor * 100 / n_cells

# Set arrow size to scale with number of edges
n_edges <- nrow(edges)

# Scake alpha by number of edges 
edge_alpha <- max(0.1, min(1, n_cells / (n_edges + 1)))

# Make colors 
edge_color <- "is_axon"
cell_color <- "layer"
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

library(plotly)

hex <- rgb(t(col2rgb(label_colors)), maxColorValue = 255)

cells$layer <- factor(
  cells$layer,
  levels = c("axon", "dendrite", "L6", "L5", "L4", "L2/3"),
  labels = c("axon", "dendrite", "L6", "L5", "L4", "L2/3")
)
edges[,edge_color] <- factor(
  edges[,edge_color],
  levels = c("axon", "dendrite", "L6", "L5", "L4", "L2/3"),
  labels = c("axon", "dendrite", "L6", "L5", "L4", "L2/3")
) 

edges_long <- data.frame(
  x = c(rbind(edges$x_start, edges$x_end, NA)),
  y = c(rbind(edges$y_start, edges$y_end, NA)),
  z = c(rbind(edges$z_start, edges$z_end, NA)),
  group = rep(edges[[edge_color]], each = 3)
)

plt <- plot_ly(
  edges_long,
  x = ~x,
  y = ~z,
  z = ~y,
  type = "scatter3d",
  mode = "lines",
  color = ~factor(group),
  colors = hex
)

# edges
plt <- plt |>
  add_trace(
    data = cells,
    x = ~x,
    y = ~y,
    z = ~z,
    type = "scatter3d",
    mode = "markers",
    color = ~factor(layer),
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



