# Return principal neurons by standard layers

This function returns the "principal" (i.e., most common) types of
neuron, from among the default known list, for each of the default known
layers.

## Usage

``` r
principal.neurons(print_nicely = FALSE)
```

## Arguments

- print_nicely:

  Logical; if TRUE, returns nothing while printing the principal neuron
  types in a nice format. If FALSE, returns a list of principal neuron
  types by layer, with layer names as keys and neuron type names as
  values (default: FALSE).

## Value

A list of principal neuron types by layer, with layer names as keys and
neuron type names as values.
