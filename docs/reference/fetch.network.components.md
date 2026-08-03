# Fetch network components

This function retrieves the components of a network object.

## Usage

``` r
fetch.network.components(
  network,
  include_arbors = FALSE,
  return_arbors = TRUE,
  verbose = TRUE
)
```

## Arguments

- network:

  Network object from which to fetch components.

- include_arbors:

  Logical indicating whether to include arbor information in the fetched
  components (can be large and computationally intensive, default =
  FALSE).

- return_arbors:

  Logical indicating whether to return the raw arbor matrix in the
  output. When FALSE and include_arbors is TRUE, the arbor matrix is
  dropped to reduce memory usage (default: TRUE). The arbor matrix can
  be large (\> 1 GB) and is used internally for computation but may not
  be needed by the user.

- verbose:

  Logical indicating whether to print a summary of the fetched
  components (default: TRUE).

## Value

A list containing the components of the network.
