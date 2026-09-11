## ---------------------------------------------------------------------------
## Scratch script: does the Hebbian update rule (DACx paper, eqs. 44 & 46)
## behave like "neurons that fire together, wire together"?
##
## Equations (sec. 2.1.6):
##   G2(g)  = Ginf + 2g + g^2/Ginf                      (eq. 41)
##   H(S)   = -1/S                                       (eq. 42)
##   O(g)   = g + g^2/Ginf                               (eq. 43)
##   dg/dt  = -(eta/Hcab) * (G2(g)*H(S) + O(g))          (eq. 44, corrected)
##   dH/dg  = Hcab / (G2(g)*H(S) + O(g))                 (eq. 46, re-derived)
##
## Variable interpretation:
##   S    = S_ij (synaptic gating trace, peaks with presynaptic spikes, capped at 1)
##   eta  = -dH_i/dt > 0 (rate of power dissipation, peaks with postsynaptic spikes)
##   Hcab = (v_cab - v_eq) * exp(L) * I_syn,i   (eq. 40, units: fW)
##          For an excitatory synapse at rest with a modestly active postsynaptic cell:
##            (v_cab - v_eq) ≈ -60 mV, exp(L) ≈ 1, I_syn,i ≈ -10 pA
##            => Hcab ≈ (-60)*1*(-10) = +600 fW
## ---------------------------------------------------------------------------

library(tidyverse)

## --- Core equations ---------------------------------------------------------

G2  <- function(g, Ginf) Ginf + 2 * g + g^2 / Ginf
Hf  <- function(S)       -1 / S
Oij <- function(g, Ginf)  g + g^2 / Ginf

dg_dt <- function(eta, Hcab, g, Ginf, S) {
  -(eta / Hcab) * (G2(g, Ginf) * Hf(S) + Oij(g, Ginf))
}

dH_dg <- function(Hcab, g, Ginf, S) {
  Hcab / (G2(g, Ginf) * Hf(S) + Oij(g, Ginf))
}

## --- Biologically realistic parameters -------------------------------------
## g0    : single-synapse conductance (nS) -- Levy & Reyes 2012 ~0.1 nS
## Ginf0 : characteristic admittance (nS) from eq. 18, typical ~2 nS for a
##         1-µm-diameter dendrite with Rm = 10 kΩ·cm², Ra = 200 Ω·cm
## Hcab  : +600 fW for an active excitatory synapse (see header)

g0    <- 0.1   # nS
Ginf0 <- 2.0   # nS

## The sign-flip threshold S* = G2(g)/O(g); check whether it falls in [0, 1].
## S_fast is capped at 1 in network::BGT, so S > 1 is unrealistic.
S_star <- G2(g0, Ginf0) / Oij(g0, Ginf0)
message(sprintf(
  "S* (sign-flip threshold) = %.1f  =>  %s within the realistic S range [0, 1].",
  S_star,
  if (S_star > 1) "NEVER reached" else "IS reachable"
))

## --- Sanity check: dg/dt * dH/dg == -eta at non-singular points -------------

check_grid <- expand.grid(
  eta  = c(0.5, 5, 50),
  Hcab = c(-600, 600),
  g    = c(0.05, 0.1, 1),
  S    = c(0.05, 0.3, 0.8)
) |>
  mutate(
    lhs = dg_dt(eta, Hcab, g, Ginf = 2, S) * dH_dg(Hcab, g, Ginf = 2, S),
    rhs = -eta,
    ok  = abs(lhs - rhs) < 1e-9
  )
n_singular <- sum(is.na(check_grid$lhs))
stopifnot(all(check_grid$ok[!is.na(check_grid$lhs)]))
message(sprintf(
  "Consistency check passed (eq. 44 * eq. 46 == -eta) at all %d non-singular points.",
  sum(!is.na(check_grid$lhs))
))

## --- Plot 1: heatmap of dg/dt over (S, eta) --------------------------------
## S restricted to [0, 1] (the realistic cap in network::BGT).
## eta in fW/ms; Hcab = +600 (excitatory) and -600 (reference).

grid1 <- expand.grid(
  S    = seq(0.01, 1, length.out = 200),
  eta  = seq(0.5, 100, length.out = 200),
  Hcab = c(-600, 600)
) |>
  mutate(
    dgdt     = dg_dt(eta, Hcab, g0, Ginf0, S),
    Hcab_lab = ifelse(Hcab > 0, "Hcab = +600 fW  (excitatory)", "Hcab = -600 fW  (reference)")
  )

p1 <- ggplot(grid1, aes(S, eta, fill = dgdt)) +
  geom_raster() +
  geom_contour(aes(z = dgdt), breaks = 0, color = "black", linewidth = 0.5) +
  scale_fill_gradient2(low = "firebrick", mid = "white", high = "steelblue", midpoint = 0) +
  facet_wrap(~Hcab_lab) +
  labs(
    x = "S_ij  (presynaptic gating trace, capped at 1 in BGT)",
    y = "eta  (fW/ms, proxy for postsynaptic spike-driven power loss)",
    fill = "dg/dt\n(nS/ms)",
    title = "Eq. 44 (corrected, realistic units): conductance change rate",
    subtitle = "Black contour = dg/dt = 0. Note: S* >> 1, so the contour never appears in this range."
  )
print(p1)

## --- Plot 2: dg/dt vs S at several eta levels, for Hcab = +600 fW ----------

grid2 <- expand.grid(
  S    = seq(0.01, 1, length.out = 300),
  eta  = c(1, 5, 20, 50),    # fW/ms
  Hcab = c(-600, 600)
) |>
  mutate(
    dgdt     = dg_dt(eta, Hcab, g0, Ginf0, S),
    Hcab_lab = ifelse(Hcab > 0, "Hcab = +600 fW  (excitatory)", "Hcab = -600 fW  (reference)"),
    eta_lab  = factor(paste0(eta, " fW/ms"))
  )

p2 <- ggplot(grid2, aes(S, dgdt, color = eta_lab)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
  geom_line(linewidth = 0.8) +
  facet_wrap(~Hcab_lab, scales = "free_y") +
  labs(
    x = "S_ij (presynaptic gating trace)",
    y = "dg/dt  (nS/ms)",
    color = "eta",
    title = "Eq. 44: conductance change rate vs presynaptic drive, at fixed eta"
  )
print(p2)

## --- Plot 3: STDP-style pre/post spike-pairing simulation ------------------
## Integrate eq. 44 forward in time for a single pre/post spike pair with
## varying relative timing. g is capped at g_max to prevent the finite-time
## blow-up (G2 ~ g^2 positive feedback) that arises without a biological
## saturation limit (which the paper does not yet specify).

simulate_dg <- function(delta_t, Hcab,
                         g0      = 0.1,   # nS
                         Ginf0   = 2.0,   # nS
                         tau_S   = 20,    # ms: presynaptic S decay
                         tau_eta = 5,     # ms: postsynaptic power-loss decay
                         S_base  = 0.05,  # baseline gating (dimensionless)
                         S_peak  = 0.8,   # gating at spike onset
                         eta_base = 0.5,  # fW/ms baseline power-dissipation rate
                         eta_peak = 50,   # fW/ms at postsynaptic spike peak
                         g_max   = 5,     # nS ceiling (biological saturation proxy)
                         dt      = 0.1,   # ms
                         t_max   = 200) { # ms
  t_pre  <- 50
  t_post <- t_pre + delta_t
  times  <- seq(0, t_max, by = dt)

  S_t   <- S_base   + ifelse(times >= t_pre,  (S_peak   - S_base)   * exp(-(times - t_pre)  / tau_S),   0)
  eta_t <- eta_base + ifelse(times >= t_post, (eta_peak - eta_base) * exp(-(times - t_post) / tau_eta), 0)

  g <- g0
  for (i in seq_along(times)) {
    g <- g + dg_dt(eta_t[i], Hcab, g, Ginf0, S_t[i]) * dt
    g <- max(g, 1e-6)
    g <- min(g, g_max)
  }
  g - g0
}

deltas <- seq(-60, 60, by = 3)

stdp <- bind_rows(
  tibble(delta_t = deltas, Hcab =  600, delta_g = map_dbl(deltas, simulate_dg, Hcab =  600)),
  tibble(delta_t = deltas, Hcab = -600, delta_g = map_dbl(deltas, simulate_dg, Hcab = -600))
) |>
  mutate(Hcab_lab = ifelse(Hcab > 0, "Hcab = +600 fW  (excitatory)", "Hcab = -600 fW  (reference)"))

p3 <- ggplot(stdp, aes(delta_t, delta_g)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
  geom_line(linewidth = 0.9) +
  geom_point(size = 1.5) +
  facet_wrap(~Hcab_lab, scales = "free_y") +
  labs(
    x = "post-spike time − pre-spike time (ms)  [0 = coincident firing]",
    y = "net change in g over the pairing event  (delta_g, nS)",
    title = "STDP-style probe of eq. 44 (corrected, realistic units, with g ceiling)",
    subtitle = "Hebbian signature: delta_g > 0 peaking near delta_t = 0"
  )
print(p3)
