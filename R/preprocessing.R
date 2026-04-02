
#' @useDynLib neurons, .registration = TRUE
#' @import R.matlab
#' @import hdf5r
#' @import reticulate
NULL

load_mat <- function(path) {
    # Read first few bytes to determine format
    con <- file(path, "rb")
    header <- readChar(con, nchars = 8, useBytes = TRUE)
    close(con)
    
    if (startsWith(header, "MATLAB 5")) {
      # old-style .mat file
      out <- R.matlab::readMat(path)[[1]]
    } else {
      # new-style HDF5 .mat file
      # Open the v7.3 MAT-file
      file <- hdf5r::H5File$new(path, mode = "r")
      out <- c(file[["/includeVector"]][,])
      
      # Release file
      file$close_all()
      
    }
    
    out
  }

#' Import raw kilosort4 data
#' 
#' Base function to import raw kilosort4 and accompanying stimulus data. The function looks for a csv file stim_data_file and looks for the following files in folder_path/kilosort4:
#' \describe{
#'    \item{spike_positions.npy}{2D array giving the x and y position of each spike}
#'    \item{spike_clusters.npy}{integer giving the cluster number of each spike}
#'    \item{spike_times.npy}{sample number the spike occurred at}
#'    \item{cluster_group.tsv}{2D array giving status of each cluster (0=noise, 1=MUA, 2=Good, 3=unsorted), hand-curated}
#'    \item{cluster_info.tsv}{2D array giving the automatic output of kilosort4. This file not needed if cluster_group.tsv has data.}
#'    \item{includeVector.mat}{MATLAB file giving whether each cluster is stimulus-responsive (1) or not (0)}
#' }
#' 
#' @param folder_path Path to the kilosort4 output files
#' @param stim_data_file Path to the stimulus event data file
#' @param recording_Fq_MATLAB Sampling frequency of the MATLAB recording
#' @param trial_time_start Time to begin trial (ms), relative to stimulus onset
#' @param trial_time_end Time to end trial (ms), relative to stimulus onset
#' @param verbose Print report on imported data?
#' @returns A list with two data frames: spikes (rows are spikes) and stim (rows are stimulus onset times)
#' @export
import.kilo4 <- function(
    folder_path,
    stim_data_file,
    recording_Fq_MATLAB = 30303,
    trial_time_start = -100,
    trial_time_end = 400,
    verbose = TRUE
  ) {
    
    # Will produce two data frames, spike (rows are spikes) and stim (rows are stimulus onset times)
    
    # Make lists of all .npy and .tsv files in the folder
    folder_path_kilo4 <- paste0(folder_path, "/kilosort4", sep = "")
    npy_files <- list.files(folder_path_kilo4, pattern = "\\.npy$", full.names = TRUE)
    tsv_files <- list.files(folder_path_kilo4, pattern = "\\.tsv$", full.names = TRUE)
    
    # Create an empty list to store data
    data_list <- list()
    
    # Files to import 
    # ... MUA = multi-unit activity, Good = single unit
    import_list <- c(
      "spike_positions",          # 2D array giving the x and y position of each spike
      "spike_clusters",           # integer giving the cluster number of each spike
      "spike_times",              # sample number the spike occurred at
      "cluster_group",            # 2D array giving status of each cluster (0=noise, 1=MUA, 2=Good, 3=unsorted), hand-curated
      "cluster_info"              # 2D array giving the automatic output of kilosort4
    )
    
    # Files to import as integers
    int_list <- c("spike_clusters")
    
    # Load .npy files 
    for (file in npy_files) {
      file_name <- gsub(paste0(folder_path_kilo4,"/"), "", file)
      file_name <- sub("\\..*", "", file_name)
      if (!(file_name %in% import_list)) next
      data_list[[file_name]] <- tryCatch({
        reticulate::import("numpy")$load(file, allow_pickle = TRUE)
      }, error = function(e) {
        cat("\n\nError loading: ", file, "\n")
        NULL
      })
    }
    
    # Load .tsv files
    for (file in tsv_files) {
      file_name <- gsub(paste0(folder_path_kilo4,"/"), "", file)
      file_name <- sub("\\..*", "", file_name)
      if (!(file_name %in% import_list)) next
      data_list[[file_name]] <- read.delim(file, sep = "\t", header = TRUE)
    }
    
    # Print report on the imported data
    if (verbose) {
      cat("\n---")
      cat("\nImported data:", gsub(sub("/[^/]*$", "", folder_path), "", file))
      for (i in 1:length(data_list)) {
        cat("\n", names(data_list)[i])
        cat(", class: ", class(data_list[[i]]))
        if (length(dim(data_list[[i]])) > 0) {
          cat(", dim: ", dim(data_list[[i]]))
          if (length(colnames(data_list[[i]])) > 0) {
            n <- min(5, dim(data_list[[i]])[2])
            cat(", col names (1st 5): ", paste(colnames(data_list[[i]])[1:n], collapse = ", "))
          }
        } else {
          cat(", length: ", length(data_list[[i]]))
        }
      }
    }
    
    # Grab stimulus response info
    stim_responsive_by_cluster <- c(load_mat(paste0(folder_path_kilo4, "/includeVector.mat")))
    # ... note: cluster_id is 0-index in the tsv files while matlab is 1-indexed, but the data imported from 
    #      this matlab file is not explicitly indexed, uses implicit row indexing (e.g., first row is the first 
    #      cluster, second row the second cluster, etc). Below, when a cluster id is pulled from the tsv file (e.g., i), 
    #      a 1 is added to the row pulled from the imported matlab data. So, everything should line up as expected. 
    
    # Convert and combine data_list to data frame 
    kilosort_spike_data <- data.frame(
      spike_clusters = data_list[["spike_clusters"]],
      spike_times = data_list[["spike_times"]],
      cluster_group = rep(NA, length(data_list[["spike_clusters"]])),
      stim_responsive = rep(NA, length(data_list[["spike_clusters"]]))
    )
    # By default, use labels from cluster_info file
    group_info <- c()
    group_id <- c()
    if ("cluster_info" %in% names(data_list)) {
      group_info <- data_list[["cluster_info"]][, "KSLabel"]
      group_id <- data_list[["cluster_info"]][, "cluster_id"]
    } else {
      group_id <- data_list[["cluster_group"]][, "cluster_id"]
    }
    ## Uncomment if using kilosort1 data
    ## group_id <- unique(kilosort_spike_data$spike_clusters)
    # However, if cluster_group has data, it is hand-curated and better, so use that instead
    if (any(colnames(data_list[["cluster_group"]]) == "KSLabel")) {
      this_col <- which(colnames(data_list[["cluster_group"]]) == "KSLabel")
      colnames(data_list[["cluster_group"]])[this_col] <- "group"
    }
    group_info_curated <- data_list[["cluster_group"]][, "group"]
    group_id_curated <- data_list[["cluster_group"]][, "cluster_id"]
    if (length(group_info_curated) > 0) {
      for (r in seq_along(group_id_curated)) {
        group_info[group_id == group_id_curated[r]] <- group_info_curated[r]
      }
    }
    for (i in group_id) {
      mask <- data_list[["spike_clusters"]] == i
      kilosort_spike_data$cluster_group[mask] <- group_info[which(group_id == i)] ## Comment out if using kilosort1 data
      kilosort_spike_data$stim_responsive[mask] <- stim_responsive_by_cluster[i + 1]
    }
    
    # Import stimulus event data from MATLAB
    stim_data <- read.csv(stim_data_file)[, c("SampleStamps_samples", "StimLength_ms")]
    colnames(stim_data) <- c(
      "time",     # timestamp of stimulus start, in samples. So "sample/recording_Fq_MATLAB*1000" gives the timestamps in ms
      "length"    # length of stimulus in ms
    )
    
    # Convert all times into ms
    kilosort_spike_data$spike_times <- kilosort_spike_data$spike_times / recording_Fq_MATLAB * 1000
    stim_data$time <- stim_data$time / recording_Fq_MATLAB * 1000
    
    # Reorder the stimulus event data frame by time 
    stim_data <- stim_data[order(stim_data$time),]
    
    # Find start and end times for each trial
    stim_data$trial_start <- rep(NA, nrow(stim_data))
    stim_data$trial_end <- rep(NA, nrow(stim_data))
    stim_data$trial_status <- rep(NA, nrow(stim_data))
    for (i in 1:nrow(stim_data)) {
      # Grab start and end times
      stim_data$trial_start[i] <- stim_data$time[i] + trial_time_start
      stim_data$trial_end[i] <- stim_data$time[i] + trial_time_end
      # Ensure trials are in order
      if (i > 1) {
        if (any(stim_data$trial_start[i] < stim_data$trial_start[1:(i-1)])) {
          stop("Trial start times are out of order.")
        }
      }
    }
    
    # Check trial validity
    stim_data$trial_status[1] <- 0
    for (i in 2:nrow(stim_data)) {
      
      # Key
      # ... 0 = good
      # ... 1 = overlapped by trailing trial
      # ... 2 = overlapped by leading trial
      # ... 3 = stimulus events in trial are from a different trial
      
      # Mark trial as good 
      stim_data$trial_status[i] <- 0
      
      # Grab indexes for trailing and leading trials 
      trailing_trials <- 1:(i-1)
      
      # All trailing trials must have ended before the current trial started
      if (any(stim_data$trial_end[trailing_trials] >= stim_data$trial_start[i])) {
        stim_data$trial_status[i] <- 1
      }
      
      # All leading trials must have started after the current trial ended
      if (i < nrow(stim_data)) {
        leading_trials <- (i + 1):nrow(stim_data)
        if (any(stim_data$trial_start[leading_trials] <= stim_data$trial_end[i])) {
          stim_data$trial_status[i] <- 2
        }
      }
      
      # Any stimulus events in the trial must be from the trial
      stim_mask <- stim_data$time >= stim_data$trial_start[i] & stim_data$time <= stim_data$trial_end[i]
      if (any(stim_mask)) {
        if (any(which(stim_mask) != i)) {
          stim_data$trial_status[i] <- 3
        }
      }
      
    }
    
    # Label trials and calculate trial-relative times
    kilosort_spike_data$trial_number <- rep(NA, nrow(kilosort_spike_data))
    kilosort_spike_data$trial_time <- rep(NA, nrow(kilosort_spike_data))
    for (i in 1:nrow(stim_data)) {
      mask <- kilosort_spike_data$spike_times > stim_data$trial_start[i] & kilosort_spike_data$spike_times <= stim_data$trial_end[i]
      kilosort_spike_data$trial_number[mask] <- i
      kilosort_spike_data$trial_time[mask] <- kilosort_spike_data$spike_times[mask] - stim_data$time[i]
    }
    
    return(
      list(
        spikes = kilosort_spike_data,
        stim = stim_data
      )
    )
    
  }

#' Preprocess kilosort4 data 
#' 
#' Function to preprocess kilosort4 data for use with neurons package functions.
#' 
#' @param trial_time_start Time to begin trial (ms), relative to stimulus onset
#' @param trial_time_end Time to end trial (ms), relative to stimulus onset
#' @param recording.folder List of paths to the kilosort4 output folders
#' @param meta_data Data frame with metadata for each recording (e.g., genotype, hemisphere), one recording per row; row names should match recording names and all columns should be covariates for later analysis
#' @param max_spikes Maximum number of total spikes for a cell to be kept
#' @param min_spikes Minimum number of total spikes for a cell to be kept
#' @param min_trials Minimum number of trials for a cell to be kept
#' @param pure_trials_only Keep only trials with no overlap?
#' @param good_cells_only Keep only cells marked as "good" in the cluster_group.tsv file?
#' @param stim_responsive_only Keep only cells marked as stimulus-responsive in the cluster_group.tsv file?
#' @param verbose Print report on imported data?
#' @returns A list with three elements: 
#'  \describe{
#'    \item{spikes}{data frame with one row per spike}
#'    \item{timeXtrial}{list of matrices with rows as sample times and columns as trials, each element a zero if no spike at that time in that trial, and a one if a spike}
#'    \item{cluster.key}{data frame with one row per neuron}
#'  }
preprocess.kilo4 <- function(
    trial_time_start = -100,
    trial_time_end = 2020,
    recording.folder = "data",
    meta_data = NULL,
    max_spikes = Inf,
    min_spikes = 0,
    min_trials = 0,
    pure_trials_only = TRUE,
    good_cells_only = TRUE,
    stim_responsive_only = TRUE,
    verbose = TRUE
  ) {
    
    recording.folders <- list.files(recording.folder, full.names = TRUE)
    n_recordings <- length(recording.folders)
    time_range <- trial_time_end - trial_time_start
    
    # Load in raw kilosort data
    kilosort.data <- list()
    kilosort.data.maxtrial <- rep(NA, n_recordings)
    names(kilosort.data.maxtrial) <- recording.folders
    for (rf in recording.folders) {
      
      # Produce two data frames, spike (rows are spikes) and stim (rows are stimulus onset times)
      kilosort.data[[rf]] <- import.kilo4(
        folder_path = rf,
        stim_data_file = paste0(rf, "/StimulusStamps.csv"),
        trial_time_start = trial_time_start,         
        trial_time_end = trial_time_end,          
        verbose = verbose
      ) 
      
      # Extract trial info
      pure_trial <- kilosort.data[[rf]]$stim$trial_status == 0
      n_trials <- length(pure_trial)
      n_pure_trials <- sum(pure_trial)
      if (pure_trials_only && n_pure_trials == 0) {
        stop(paste0("No pure trials found, neuron recording", rf))
      }
      kilosort.data.maxtrial[rf] <- n_trials
      if (pure_trials_only) kilosort.data.maxtrial[rf] <- n_pure_trials
      
      # Clean spike data 
      # ... tag impure (overlapping) trials
      if (pure_trials_only) {
        for (t in 1:n_trials) {
          if (!pure_trial[t]) {
            t_mask <- kilosort.data[[rf]]$spikes$trial_number == t
            kilosort.data[[rf]]$spikes$trial_number[t_mask] <- NA
          }
        }
      }
      # ... remove spikes outside of the trial time range and impure trials
      no_trial_spikes <- is.na(kilosort.data[[rf]]$spikes$trial_number)
      kilosort.data[[rf]]$spikes <- kilosort.data[[rf]]$spikes[!no_trial_spikes,]
      
      # ... renumber trials 
      if (pure_trials_only) {
        new_row_names <- 1:n_trials
        if (n_pure_trials < n_trials) {
          new_row_names <- c(1:n_pure_trials, (n_pure_trials + 1):n_trials)
        }
        reorder_idx <- c(which(pure_trial), which(!pure_trial))
        rownames(kilosort.data[[rf]]$stim)[reorder_idx] <- new_row_names
        old_nums <- which(pure_trial)
        for (i in 1:n_pure_trials) {
          t_mask <- kilosort.data[[rf]]$spikes$trial_number == old_nums[i]
          kilosort.data[[rf]]$spikes$trial_number[t_mask] <- i
        }
      }
      
    }
    
    # Define mask for good cells 
    good_mua <- function(df) {
      out <- TRUE
      if (good_cells_only) out <- out & df$cluster_group == "good"
      if (stim_responsive_only) out <- out & df$stim_responsive == 1
      return(out)
    }
    ## Uncomment if using kilosort1 data
    ## good_mua <- function(df) df$stim_responsive == 1
    
    # Set up list to hold spike rasters 
    # ... make names
    raster_names <- c()
    matrix_col_num <- c()
    for (rf in recording.folders) {
      # Find names of good cells in this recording
      kilosort_data <- kilosort.data[[rf]]$spikes
      single_cells_mask <- good_mua(kilosort_data)
      cell_numbers <- sort(unique(kilosort_data$spike_clusters[single_cells_mask]))
      if (length(cell_numbers) == 0) next
      # Combine with recording name 
      raster_names <- c(raster_names, paste0(sub(paste0(recording.folder, "/"), "", rf), "_", cell_numbers))
      # Grab max trial number 
      matrix_col_num <- c(matrix_col_num, rep(kilosort.data.maxtrial[rf], length(cell_numbers)))
    }
    # ... setup the shell of the list
    kilosort_data_parsed_timeXtrial <- as.list(
      rep(
        list(matrix(0, nrow = time_range, ncol = kilosort.data.maxtrial[1])), 
        length(raster_names) 
      )
    )
    names(kilosort_data_parsed_timeXtrial) <- raster_names
    names(matrix_col_num) <- raster_names
    # ... resize the matrices in the list
    for (rn in raster_names) {
      kilosort_data_parsed_timeXtrial[[rn]] <- matrix(0, nrow = time_range, ncol = matrix_col_num[rn])
      colnames(kilosort_data_parsed_timeXtrial[[rn]]) <- paste0("trial_", 1:ncol(kilosort_data_parsed_timeXtrial[[rn]]))
    }
    # ... make new cell numbers
    cell_numbers_good <- 1:length(raster_names)
    names(cell_numbers_good) <- raster_names
    
    # Extract and preprocess good neurons 
    kilosort_data_parsed_spikes <- data.frame()
    for (rf in recording.folders) {
      
      # Grab spike data
      kilosort_data <- kilosort.data[[rf]]$spikes
      
      # Take only good multi-unit clusters which plausibly represent a single neuron
      single_cells_mask <- good_mua(kilosort_data)
      cell_numbers <- sort(unique(kilosort_data$spike_clusters[single_cells_mask]))
      
      # For each cell, extract data as matrices with rows as sample times, columns as trials
      for (cn in cell_numbers) {
        # ... grab raster name 
        rn <- paste0(sub(paste0(recording.folder, "/"), "", rf), "_", cn)
        # ... grab spikes which belong to that cell and are in a valid trial
        spike_idx <- which(kilosort_data$spike_clusters == cn)
        # For each spike ...
        for (i in spike_idx) {
          # ... find its time and trial
          spike_time <- round(kilosort_data$trial_time[i], 0) 
          trial_num <- kilosort_data$trial_number[i]
          # ... and represent it as "1" in the matrix
          kilosort_data_parsed_timeXtrial[[rn]][spike_time - trial_time_start, trial_num] <- 1
        }
      }
      
      # Build single "sparse" raster with extra data columns
      # Subset the raw kilosort_data data to include only spikes which belong to some good cell and are in a valid trial
      single_notna <- single_cells_mask & !is.na(kilosort_data$trial_number)
      if (sum(single_notna) > 0) {
        kilosort_data_parsed_spikes_good <- kilosort_data[single_notna, ]
        original_cluster_nums <- kilosort_data$spike_clusters[single_notna]
        # ... rename the columns
        colnames(kilosort_data_parsed_spikes_good) <- c("cell", "sample", "cluster_group", "stim_responsive", "trial", "time_in_ms")
        # ... reorder and subset columns
        kilosort_data_parsed_spikes_good <- kilosort_data_parsed_spikes_good[, c("trial", "sample", "cell", "time_in_ms")]
        # ... add recording name, cluster, and covariate information
        recordingname <- sub(paste0(recording.folder, "/"), "", rf)
        kilosort_data_parsed_spikes_good$recording_name <- recordingname
        kilosort_data_parsed_spikes_good$cluster <- original_cluster_nums
        if (!is.null(meta_data)) {
          for (covariate in colnames(meta_data)) kilosort_data_parsed_spikes_good[[covariate]] <- meta_data[recordingname, covariate]
        }
        # ... replace cell numbers with new cell numbers
        replaced <- rep(FALSE, nrow(kilosort_data_parsed_spikes_good))
        for (cn in cell_numbers) { 
          rn <- paste0(sub(paste0(recording.folder, "/"), "", rf), "_", cn)
          cell_mask <- !replaced & kilosort_data_parsed_spikes_good$cell == cn
          kilosort_data_parsed_spikes_good$cell[cell_mask] <- cell_numbers_good[rn]
          replaced <- replaced | cell_mask
        }
        kilosort_data_parsed_spikes <- rbind(
          kilosort_data_parsed_spikes,
          kilosort_data_parsed_spikes_good
        )
      }
    }
    
    # Make cluster key and check for low-spike or low-trial cells
    cell_nums <- unique(kilosort_data_parsed_spikes$cell)
    recording_name_col <- c()
    cell_col <- c()
    cluster_col <- c()
    num_of_spikes_col <- c()
    num_of_trials_col <- c()
    low_mask <- FALSE
    names(cell_numbers_good) <- NULL
    new_cell_counter <- 0
    saved_masks <- list()
    for (c in cell_nums) {
      # Make mask for this cell
      mask <- kilosort_data_parsed_spikes$cell == c
      # Grab recording name
      recording_name <- unique(kilosort_data_parsed_spikes$recording_name[mask])
      # Grab cluster number
      cluster <- unique(kilosort_data_parsed_spikes$cluster[mask])
      # Run checks
      if (length(recording_name) != 1) stop("Recording name not unique")
      if (length(cluster) != 1) stop("Cluster not unique")
      if (!is.null(meta_data)) {
        for (covariate in colnames(meta_data)) {
          this_covariate <- unique(kilosort_data_parsed_spikes[mask,covariate])
          if (length(this_covariate) != 1) stop(paste0(covariate, " not unique"))
        }
      }
      # Check number of spikes and number of trials
      num_of_spikes <- sum(mask)
      num_of_trials <- length(unique(kilosort_data_parsed_spikes$trial[mask]))
      if (num_of_spikes >= min_spikes && num_of_trials >= min_trials && num_of_spikes <= max_spikes) {
        # Set new cell number 
        new_cell_counter <- new_cell_counter + 1
        # Extend columns 
        recording_name_col <- c(recording_name_col, recording_name)
        cell_col <- c(cell_col, new_cell_counter)
        cluster_col <- c(cluster_col, cluster)
        num_of_spikes_col <- c(num_of_spikes_col, num_of_spikes)
        num_of_trials_col <- c(num_of_trials_col, num_of_trials)
        # Save mask to update cell numbers in the spike data
        saved_masks[[as.character(new_cell_counter)]] <- mask
      } else {
        # Add to low mask
        low_mask <- low_mask | mask
      }
      
    }
    cluster_key <- data.frame(
      recording.name = recording_name_col,
      cell = cell_col,
      cluster = cluster_col,
      num.of.spikes = num_of_spikes_col,
      num.of.responsive.trials = num_of_trials_col
    )
    cluster_key <- cbind(cluster_key, meta_data[as.character(recording_name_col), ])
    rownames(cluster_key) <- NULL
    
    # Update cell numbers in the spike data
    for (c in 1:new_cell_counter) {
      mask <- saved_masks[[as.character(c)]]
      if (is.null(mask)) stop("Saved mask not found")
      kilosort_data_parsed_spikes$cell[mask] <- c
    }
    # Remove low-spike or low-trial cells from the spike data
    kilosort_data_parsed_spikes <- kilosort_data_parsed_spikes[!low_mask, ]
    
    return(
      list(
        spikes = kilosort_data_parsed_spikes,
        timeXtrial = kilosort_data_parsed_timeXtrial,
        cluster.key = cluster_key
      )
    )
    
  }

#' Summarize cluster key by covariates
#' 
#' Function to summarize the \code{cluster.key} output of \code{preprocess.kilo4} by covariates.
#' 
#' @param key The \code{cluster.key} data frame output of \code{preprocess.kilo4}
#' @param covariate_list List of covariates to summarize
#' @returns A data frame with one row per combination of covariates, and columns giving the number of cells, mean number of spikes, and mean number of trials for that combination
summarize.cluster.key <- function(
    key,
    covariate_list
  ) {
    
    # Get all combinations of covariates
    unique_covariates <- list()
    for (covariate in covariate_list) {
      unique_covariates[[covariate]] <- unique(key[[covariate]])
    }
    covariate_combos <- expand.grid(unique_covariates)
    
    # Initialize variables to hold summary stats
    n_cells <- rep(NA, nrow(covariate_combos))
    mean_spikes <- rep(NA, nrow(covariate_combos))
    mean_trials <- rep(NA, nrow(covariate_combos))
    
    # Loop through combinations and get summary stats
    for (cv in 1:nrow(covariate_combos)) {
      # Get combination of covariates
      combo <- covariate_combos[cv, ]
      
      # Subset cluster.key for this combination
      mask <- TRUE
      for (covariate in covariate_list) {
        mask <- mask & key[[covariate]] == combo[[covariate]]
      }
      subset_key <- key[mask, ]
      
      # If any cells in this combination, compute and save stats
      if (nrow(subset_key) > 0) {
        n_cells[cv] <- nrow(subset_key)
        mean_spikes[cv] <- round(mean(subset_key$num.of.spikes),1)
        mean_trials[cv] <- round(mean(subset_key$num.of.responsive.trials),1)
      }
    }
    
    # Return results 
    return(data.frame(covariate_combos, n_cells, mean_spikes, mean_trials))
    
  }