# Fetch network components

This function retrieves the components of a network object.

## Usage

``` r
fetch.network.components(network, include_arbors = FALSE, verbose = TRUE)
```

## Arguments

- network:

  Network object from which to fetch components.

- include_arbors:

  Logical indicating whether to include arbor information in the fetched
  components (can be large and computationally intensive, default =
  FALSE).

- verbose:

  Logical indicating whether to print a summary of the fetched
  components (default: TRUE).

## Value

A list containing the components of the network.
