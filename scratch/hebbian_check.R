## =============================================================================
## scratch/hebbian_check.R
## Visualizing the Hebbian learning rule in DACx (paper sec. 2.1.6)
##
## PURPOSE
## -------
## The DACx paper derives a local Hebbian-style learning rule from the Baum-
## Eagon inequality (eq. 3) and the synaptic electrophysiology model (sec. 2.1.5).
## As of Sep 2026, this rule has NOT yet been implemented in the C++ source
## (src/DACx.cpp). This script checks whether the rule, as written, is consistent
## with "neurons that fire together, wire together" before implementation.
##
## HOW TO RUN
## ----------
## 1. library(DACx)   -- or install with devtools::install()
## 2. source("scratch/hebbian_check.R")
## Plots are printed to the active device. No network simulation is needed.
## The G_inf and g_syn values used here were derived offline from the EI-loops
## vignette network (see "Biologically realistic parameters" section below).
##
## BACKGROUND: THE DACx LEARNING RULE (sec. 2.1.6)
## -------------------------------------------------
## DACx neurons minimize electrical power H. The Baum-Eagon inequality implies:
##
##   (dH/dv)(dv/dt) <= 0                                              (eq. 34)
##
## Expanding through the chain rule for synaptic conductance g_ij (the spike-
## dependent conductance of the synapse from presynaptic cell j onto
## postsynaptic cell i) and rearranging gives the update rule:
##
##   dg_ij/dt = -(eta / Hcab_ij) * (G2_ij(g_ij)*H(S_ij) + O_ij)    (eq. 44)
##
## where:
##   G2_ij(g) = G_inf_ki(j) + 2g + g^2/G_inf_ki(j)     (eq. 41, "hyperbolic slope")
##   H(S)     = -1/S                                     (eq. 42, "hyperbolic decay")
##   O_ij(g)  = g + g^2/G_inf_ki(j)                      (eq. 43, "hyperbolic offset")
##
## and the auxiliary quantities are:
##   S_ij     = synaptic gating trace (eq. 30):
##                S_ij = S_fast_ij + phi * max(S_slow_ij - 1, 0)
##              S_fast is incremented by 1 at each presynaptic spike and decays
##              with tau_syn_fast (fast, distance-independent).
##              S_slow is also incremented by 1 at each presynaptic spike and
##              decays with tau_syn_slow * L_norm (slow, distance-stretched).
##              IMPORTANT: S_fast is capped at 1, but S_slow is NOT capped,
##              so S_ij can exceed 1 when the presynaptic cell fires repeatedly.
##              See network::BGT (src/DACx.cpp lines 3005-3006).
##
##   eta      = -dH_i/dt > 0, the rate at which total somatic power is falling.
##              This is largest during and just after a postsynaptic spike
##              (when I_spike dominates dH/dv). Proxy for postsynaptic activity.
##
##   Hcab_ij  = (v_cab_ij - v_eq_ij) * exp(L_ki(j)) * I_syn,i       (eq. 40)
##              where v_cab_ij is the passive-cable local voltage at synapse j
##              (eq. 27), v_eq_ij is the synaptic equilibrium potential,
##              L_ki(j) is the log-attenuation to the soma (eq. 19), and
##              I_syn,i is the TOTAL effective synaptic current at the soma of i
##              summed over all presynaptic inputs (NOT just from j).
##              Units: mV * nS * pA = fW (femtowatts).
##
##   G_inf_ki(j) = characteristic admittance of the dendritic segment holding
##                 synapse j (eq. 18): G_inf = pi*d^2 / (4*Ra*lambda),
##                 lambda = sqrt(d*Rm/(4*Ra)). Units: nS.
##
## From eq. 45 (dH_i/dg_ij = -eta * dt/dg_ij), eq. 46 follows directly:
##   dH_i/dg_ij = Hcab_ij / (G2_ij(g_ij)*H(S_ij) + O_ij)           (eq. 46)
##
## SIGN CORRECTION HISTORY (eq. 44)
## ----------------------------------
## The original derivation in the paper had a sign error: the bracket was written
## as (G2*H(S) - O) rather than (G2*H(S) + O). The corrected form is
## -(eta/Hcab)*(G2*H(S) + O) as implemented here. With H(S) = -1/S:
##   bracket = -G2/S + O
## This crosses zero at the threshold:
##   S* = G2(g) / O(g)
## Below S*: bracket < 0, so sign(dg/dt) = sign(eta/Hcab) = sign(Hcab) [since eta>0].
## Above S*: bracket > 0, so sign(dg/dt) = -sign(Hcab).
##
## SIGN OF Hcab FOR AN EXCITATORY SYNAPSE (outward-positive convention)
## ---------------------------------------------------------------------
## v_cab - v_eq: for excitatory (v_eq ~ 0 mV), v_cab ~ v_rest ~ -70 mV => ~-60 mV
## exp(L):       always positive
## I_syn,i:      excitatory (inward) current is negative in outward-positive convention
##               => I_syn,i ~ -10 pA for a modestly active postsynaptic cell
## => Hcab ~ (-60) * 1 * (-10) = +600 fW  (POSITIVE for excitatory synapse)
##
## With Hcab > 0: below S*, dg/dt > 0 (LTP); above S*, dg/dt < 0 (LTD).
##
## BIOLOGICALLY REALISTIC PARAMETERS (from EI-loops vignette network)
## -------------------------------------------------------------------
## Parameters were extracted using fetch.cell.type.params("spiny stellate")
## after building the tutorial_EI_loops.Rmd network:
##
##   g_syn (ss->ss)  = 0.4 nS          [cell type parameter]
##   g_syn (PV->ss)  = 4.0 nS          [cell type parameter]
##   Rm              = 20 kOhm.cm^2    [cell type cable parameter]
##   Ri              = 100 Ohm.cm      [cell type cable parameter]
##   dendrite d      = 0.1 to 0.4 um   [basal_radius=0.4, min_radius=0.1]
##
##   G_inf (eq. 18)  = 0.03 to 0.28 nS over the diameter range above.
##                     Use 0.15 nS as a representative mid-dendritic value.
##
##   => S* = G2(0.4, 0.15) / O(0.4, 0.15) ~ 1.38
##
## CRITICAL FINDING: S* ~ 1.38 is reachable in practice because S_emit can
## exceed 1 when S_slow > 1 (from repeated presynaptic bursting). This means
## the sign-flip threshold IS relevant to network dynamics.
##
## For ss->ss synapses (g=0.4, Ginf range 0.03-0.28): S* ranges ~1.1 to 1.7.
## For PV->ss synapses (g=4.0, same Ginf):             S* ranges ~1.0 to 1.1.
##
## (Previously, G_inf was mistakenly assumed to be ~2 nS, giving S* ~ 21
## which is unreachable. The correction came from using actual cable parameters.)
##
## WHAT EACH PLOT SHOWS
## --------------------
## Plot 1 (heatmap): dg/dt over the (S_ij, eta) plane. The zero-contour at
##   S* ~ 1.38 is visible. For Hcab=+600: LTP (blue) left of contour (low
##   presynaptic drive), LTD (red) right (high/sustained presynaptic drive).
##   Note the strong 1/S amplification near S=0 that dominates the color scale.
##
## Plot 2 (line plot): dg/dt vs S at fixed eta values. Clearly shows the sign
##   flip at S*, and that the magnitude is dominated by 1/S near S=0.
##
## Plot 3 (instantaneous probe): To cleanly separate the timing-dependence from
##   the 1/S blow-up that corrupted earlier time-integration attempts, this plot
##   computes dg/dt at the MOMENT the pre- and post-synaptic pulses first meet:
##
##   Pre fires first (delta_t >= 0): at t_post, S has decayed to exp(-delta_t/tau_S)
##     => dg/dt = dg_dt(eta_peak, Hcab, g0, Ginf0, S_peak*exp(-delta_t/tau_S))
##
##   Post fires first (delta_t < 0): at t_pre, eta has decayed to eta_peak*exp(delta_t/tau_eta)
##     => dg/dt = dg_dt(eta_peak*exp(delta_t/tau_eta), Hcab, g0, Ginf0, S_peak)
##
##   KEY FINDING: For Hcab=+600 (excitatory), the pre-then-post side (solid line)
##   shows dg/dt INCREASING exponentially as delta_t grows. This is because as S
##   decays, 1/S amplifies the update: the rule potentiates MORE when the pre
##   spike was a LONG time ago, not when it was recent. This is ANTI-Hebbian in
##   the temporal sense for the causal ordering.
##
##   The post-then-pre side (dashed) shows the expected Hebbian decay (proportional
##   to remaining eta), but its magnitude is tiny.
##
## OPEN QUESTION
## -------------
## The root cause is H(S) = -1/S (eq. 42). This makes the learning rate
## INVERSELY proportional to presynaptic drive: strongest when S is near zero
## (little/no recent presynaptic activity), weakest at peak S (strong recent
## presynaptic activity). The temporal profile for pre-before-post is therefore
## anti-Hebbian.
##
## A possible fix: replace H(S) = -1/S with H(S) = S (or H(S) = -S or similar
## monotone form) so that the learning rate grows with presynaptic drive. This
## would change the mathematical derivation (eqs. 39-44) and may alter what
## biological principle the rule implements.
##
## Alternatively: in the full network, Hcab_ij depends on I_syn,i (the global
## postsynaptic current, not just the jth synapse), which may introduce an
## additional coincidence-dependent modulation not captured here (where Hcab is
## held fixed). This coupling is a candidate for further exploration.
##
## RELATED CHANGES MADE IN THIS SESSION (Sep 2026)
## ------------------------------------------------
## - src/DACx.cpp line 3101: fixed sign of eq. 26. Was:
##     v_syn = v_syn_cable + v_syn_fast + v_syn_slow   (wrong: + instead of -)
##   Corrected to:
##     v_syn = v_syn_cable - (v_syn_fast + v_syn_slow)
## - vignettes/tutorial_EI_loops.Rmd: removed deleted parameter dendrite_velocity
##   from modify.cell.type() call (parameter removed from DACx.R when MET-based
##   dendritic propagation delay replaced the old velocity-based approach).
## =============================================================================

library(tidyverse)

## --- Core equations (DACx paper sec. 2.1.6) ---------------------------------

G2  <- function(g, Ginf) Ginf + 2 * g + g^2 / Ginf       # eq. 41
Hf  <- function(S)       -1 / S                            # eq. 42
Oij <- function(g, Ginf)  g + g^2 / Ginf                  # eq. 43

dg_dt <- function(eta, Hcab, g, Ginf, S) {
  -(eta / Hcab) * (G2(g, Ginf) * Hf(S) + Oij(g, Ginf))   # eq. 44 (corrected sign)
}

dH_dg <- function(Hcab, g, Ginf, S) {
  Hcab / (G2(g, Ginf) * Hf(S) + Oij(g, Ginf))             # eq. 46 (re-derived from corrected 44)
}

## --- Biologically realistic parameters --------------------------------------

g0    <- 0.4    # nS  (ss->ss g_syn, from EI-loops network cell type)
Ginf0 <- 0.15   # nS  (mid-dendritic G_inf, from eq. 18 + cable params above)

S_star <- G2(g0, Ginf0) / Oij(g0, Ginf0)
message(sprintf(
  "S* (dg/dt sign-flip threshold) = %.2f  [reachable: S_emit can exceed 1 via uncapped S_slow]",
  S_star
))

## --- Sanity check: eq. 45 identity dg/dt * dH/dg == -eta -------------------
## Verifies eqs. 44 and 46 are mutually consistent. NAs arise at the exact
## zero-crossing (S = S*) where the bracket = 0 and dH/dg has a pole; these
## are singular by construction, not bugs.

check_grid <- expand.grid(
  eta  = c(0.5, 5, 50),
  Hcab = c(-600, 600),
  g    = c(0.05, 0.1, 1),
  S    = c(0.05, 0.3, 0.8)
) |>
  mutate(
    lhs = dg_dt(eta, Hcab, g, Ginf0, S) * dH_dg(Hcab, g, Ginf0, S),
    rhs = -eta,
    ok  = abs(lhs - rhs) < 1e-9
  )
stopifnot(all(check_grid$ok[!is.na(check_grid$lhs)]))
message(sprintf(
  "Consistency check passed (dg/dt * dH/dg == -eta) at all %d non-singular points.",
  sum(!is.na(check_grid$lhs))
))

## --- Plot 1: heatmap of dg/dt over (S, eta) ---------------------------------
## S ranges to 3 because S_emit = S_fast + tA*max(S_slow-1,0) can exceed 1.
## The zero-contour at S* ~ 1.38 is visible. Hcab > 0: LTP left, LTD right.
## The 1/S blow-up near S=0 dominates the color scale.

grid1 <- expand.grid(
  S    = seq(0.01, 3, length.out = 200),
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
    x        = "S_ij  (presynaptic gating trace; S_emit can exceed 1 via uncapped S_slow)",
    y        = "eta  (fW/ms, proxy for postsynaptic spike-driven power loss)",
    fill     = "dg/dt\n(nS/ms)",
    title    = "Eq. 44 (corrected): conductance change rate over (S, eta)",
    subtitle = sprintf(
      "Black contour = dg/dt = 0 at S* = %.2f.  Hcab > 0: LTP (blue) left of contour, LTD (red) right.",
      S_star
    )
  )
print(p1)

## --- Plot 2: dg/dt vs S at fixed eta values ---------------------------------
## Shows the sign flip at S* and that the 1/S term dominates near S=0.
## For Hcab=+600: LTP for S < S*, LTD for S > S*; magnitude decreases with S.

grid2 <- expand.grid(
  S    = seq(0.01, 3, length.out = 300),
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
  geom_vline(xintercept = S_star, linetype = "dotted", color = "black", linewidth = 0.4) +
  geom_line(linewidth = 0.8) +
  facet_wrap(~Hcab_lab, scales = "free_y") +
  labs(
    x        = "S_ij (presynaptic gating trace)",
    y        = "dg/dt  (nS/ms)",
    color    = "eta",
    title    = "Eq. 44: conductance change rate vs presynaptic drive",
    subtitle = sprintf("Dotted vertical line = S* = %.2f (sign-flip threshold)", S_star)
  )
print(p2)

## --- Plot 3: instantaneous learning-rate-at-coincidence probe ---------------
## Time-integration is dominated by the 1/S blow-up: even tiny remaining S
## produces enormous dg/dt via -G2/S, so the cumulative integral saturates
## regardless of timing. To isolate timing-dependence cleanly, we instead
## compute dg/dt at the SINGLE MOMENT the two pulses first overlap:
##
##   Pre fires first (delta_t >= 0):
##     At t = t_post, S has decayed from S_peak to S_peak*exp(-delta_t/tau_S).
##     eta is at full peak eta_peak (post spike just arrived).
##     => dg/dt = dg_dt(eta_peak, Hcab, g0, Ginf0, S_peak*exp(-delta_t/tau_S))
##
##   Post fires first (delta_t < 0):
##     At t = t_pre, S = S_peak (pre spike just arrived).
##     eta has decayed to eta_peak*exp(delta_t/tau_eta)  [delta_t negative].
##     => dg/dt = dg_dt(eta_peak*exp(delta_t/tau_eta), Hcab, g0, Ginf0, S_peak)
##
## tau_S   = tau_syn_fast = 2 ms (ss->ss, from cell type params)
## tau_eta = 5 ms (phenomenological postsynaptic spike-power-loss decay)
## S_peak  = 1   (one BGT spike increments S_fast by 1, then S_fast is capped at 1)
## eta_peak = 50 fW/ms (near-spike peak power-dissipation rate)
##
## KEY RESULT (Hcab = +600, excitatory):
##   Post-then-pre (dashed, delta_t < 0): dg/dt is tiny, decaying smoothly as
##     |delta_t| grows. Standard Hebbian decay -- but magnitude small.
##   Pre-then-post  (solid, delta_t > 0): dg/dt INCREASES exponentially with
##     delta_t, because S = S_peak*exp(-delta_t/tau_S) shrinks and 1/S grows.
##     => The rule potentiates MOST when the pre spike was a LONG time ago.
##        This is ANTI-Hebbian in the causal (pre-before-post) temporal sense.
##
## The root cause is H(S) = -1/S (eq. 42). If H(S) were instead proportional
## to S, the learning rate would grow WITH presynaptic drive, producing a
## conventional STDP profile. Whether this is an error in eq. 42 or an
## intentional choice with a different biological rationale is the open question.

tau_S    <- 2.0    # ms  (tau_syn_fast, ss->ss)
tau_eta  <- 5.0    # ms  (postsynaptic power-loss decay)
S_peak   <- 1.0    # single spike increments S_fast by 1 (then capped at 1 in BGT)
eta_peak <- 50.0   # fW/ms

deltas_fine <- seq(-20, 20, by = 0.25)

rate_at_coincidence <- function(delta_t, Hcab) {
  if (delta_t >= 0) {
    S_at_post <- S_peak * exp(-delta_t / tau_S)
    dg_dt(eta_peak, Hcab, g0, Ginf0, S_at_post)
  } else {
    eta_at_pre <- eta_peak * exp(delta_t / tau_eta)   # delta_t < 0
    dg_dt(eta_at_pre, Hcab, g0, Ginf0, S_peak)
  }
}

stdp_instant <- bind_rows(
  tibble(delta_t = deltas_fine, Hcab =  600,
         dgdt_at_coincidence = map_dbl(deltas_fine, rate_at_coincidence, Hcab =  600)),
  tibble(delta_t = deltas_fine, Hcab = -600,
         dgdt_at_coincidence = map_dbl(deltas_fine, rate_at_coincidence, Hcab = -600))
) |>
  mutate(
    Hcab_lab = ifelse(Hcab > 0, "Hcab = +600 fW  (excitatory)", "Hcab = -600 fW  (reference)"),
    side     = ifelse(delta_t >= 0, "pre then post  (solid)", "post then pre  (dashed)")
  )

p3 <- ggplot(stdp_instant, aes(delta_t, dgdt_at_coincidence, linetype = side)) +
  geom_hline(yintercept = 0, color = "grey50", linetype = "dashed") +
  geom_vline(xintercept = 0, color = "grey50", linetype = "dashed") +
  geom_line(linewidth = 0.9) +
  facet_wrap(~Hcab_lab, scales = "free_y") +
  scale_linetype_manual(values = c("pre then post  (solid)"  = "solid",
                                   "post then pre  (dashed)" = "dashed")) +
  labs(
    x        = "post-spike time \u2212 pre-spike time  (ms,  0 = coincident firing)",
    y        = "dg/dt at moment of coincidence  (nS/ms)",
    linetype = NULL,
    title    = "Instantaneous learning rate at coincidence  (eq. 44, corrected)",
    subtitle = paste0(
      "For Hcab = +600 fW (excitatory): solid line rises exponentially with delta_t\n",
      "=> 1/S amplification makes rule ANTI-Hebbian for the pre-before-post ordering.\n",
      "Open question: should H(S) = -1/S (eq. 42) be replaced by a form that grows with S?"
    )
  )
print(p3)
