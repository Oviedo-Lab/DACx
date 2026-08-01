# DACx Documentation Updates Needed

## Summary
The following functions have implementation changes that are not reflected in their documentation comments.

---

## 1. `fetch.network.components()` — Lines 678-734

### Issue
The documentation does not include the `return_arbors` parameter.

### Current Documentation (Lines 670-677)
```r
#' @param network Network object from which to fetch components.
#' @param include_arbors Logical indicating whether to include arbor information in the fetched components (can be large and computationally intensive, default = FALSE).
#' @param verbose Logical indicating whether to print a summary of the fetched components (default: TRUE).
#' @return A list containing the components of the network.
```

### Actual Function Signature (Lines 678-683)
```r
fetch.network.components <- function(
    network, 
    include_arbors = FALSE,
    return_arbors  = TRUE,
    verbose        = TRUE
  )
```

### Required Update
Add documentation for the `return_arbors` parameter:
```r
#' @param return_arbors Logical indicating whether to return the raw arbor matrix in the output. When FALSE and include_arbors is TRUE, the arbor matrix is dropped to reduce memory usage (default: TRUE). The arbor matrix can be large (> 1 GB) and is used internally for computation but may not be needed by the user.
```

### Implementation Notes
- The function uses `return_arbors` to optionally drop the `arbors` field from the returned list (see lines 704-706)
- This is documented in the code comments as being useful for keeping the returned object cacheable by knitr
- The parameter only has an effect when `include_arbors = TRUE`

---

## 2. `plot.network()` — Lines 794-810

### Issue
Default value for `soma_size_factor` parameter differs between documentation and implementation.

### Current Documentation (Line 773)
```r
#'  soma_size_factor = 3.0,
```

### Actual Function Default (Line 807)
```r
    soma_size_factor      = 0.5,
```

### Required Update
Change the usage section (line 773) to match the actual default:
```r
#'  soma_size_factor = 0.5,
```

Also update the documentation comment at line 789:
```r
#' @param soma_size_factor Numeric value controlling how cell size in the plot scales to the number of cells (default: 0.5).
```

### Implementation Notes
- The actual default used in the function is 0.5 (line 807)
- This is then multiplied by 2 when plotting in 3D (line 1036)
- The parameter is used in the calculation at line 1037

---

## 3. `plot.network()` — Return value documentation (Line 790)

### Issue
Default value for `return_plot` parameter differs between documentation and implementation.

### Current Documentation (Line 790)
```r
#' @param return_plot Logical indicating whether to return the ggplot object or print the plot directly (default: TRUE).
```

Note: Documentation says default is TRUE (return the plot).

### Actual Function Default (Line 808)
```r
    return_plot           = TRUE,
```

✓ **This is CORRECT** — no update needed. The documentation and implementation match (both TRUE).

However, the documentation at line 774 shows `return_plot = FALSE`, which is inconsistent.

### Required Update
Change the usage section (line 774) to match the actual default:
```r
#'  return_plot = TRUE,
```

---

## 4. `plot.network()` — Return value documentation (Line 791)

### Issue
Default value for `return_cell_arbor_idx` parameter differs between documentation and implementation.

### Current Documentation (Line 791)
```r
#' @param return_cell_arbor_idx Logical indicating whether to return the soma_mask and arbor_idx used for plotting or not (default: TRUE).
```

### Actual Function Default (Line 809)
```r
    return_cell_arbor_idx = TRUE,
```

✓ **This is CORRECT** — no update needed.

However, the documentation at line 775 shows `return_cell_arbor_idx = FALSE`, which is inconsistent.

### Required Update
Change the usage section (line 775) to match the actual default:
```r
#'  return_cell_arbor_idx = TRUE
```

---

## Summary of Required Changes

| Function | Line(s) | Parameter | Change |
|----------|---------|-----------|--------|
| `fetch.network.components()` | 670-677 | `return_arbors` | Add missing parameter documentation |
| `plot.network()` | 773 | `soma_size_factor` | Change default from 3.0 to 0.5 in usage section |
| `plot.network()` | 774 | `return_plot` | Change default from FALSE to TRUE in usage section |
| `plot.network()` | 775 | `return_cell_arbor_idx` | Change default from FALSE to TRUE in usage section |
| `plot.network()` | 789 | `soma_size_factor` | Add/clarify default value (0.5) in parameter description |

