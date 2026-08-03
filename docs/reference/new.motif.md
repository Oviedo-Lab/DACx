# Initialize network (circuit) motif

This function initializes a new motif object with specified parameters.
Motifs are used for building networks of interconnected neurons. They
are recipes for building internode projections within a neural network.
They are "columnar", in the sense that they are repeated across cortical
columns.

## Usage

``` r
new.motif(motif_name = "not_provided", hemi = "both")
```

## Arguments

- motif_name:

  Character string giving name of the motif (default: "not_provided").

- hemi:

  Hemisphere to which the motif applies: use 0 or "left" for left, 1 or
  "right" for right, and -1, "all", or "both" for left and right.

## Value

A new motif object.
