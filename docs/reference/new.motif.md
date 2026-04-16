# Initialize network (circuit) motif

This function initializes a new motif object with specified parameters.
Motifs are used for building networks of interconnected neurons. They
are recipes for building internode projections within a neural network.
They are "columnar", in the sense that they are repeated across cortical
columns.

## Usage

``` r
new.motif(motif_name = "not_provided")
```

## Arguments

- motif_name:

  Character string giving name of the motif (default: "not_provided").

## Value

A new motif object.
