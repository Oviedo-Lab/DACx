# Fetch cell type parameters

This function returns the parameters for a named cell type in a list.
It's just a wrapper for the Rcpp-exported `fetch_cell_type_params`
function.

## Usage

``` r
fetch.cell.type.params(type_name)
```

## Arguments

- type_name:

  Character string giving name of the cell type, e.g. "pyramidal", "PV",
  "SST", etc.

## Value

List of parameters for the named cell type.
