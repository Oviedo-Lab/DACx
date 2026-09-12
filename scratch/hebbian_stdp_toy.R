## =============================================================================
## scratch/hebbian_stdp_toy.R
## A 2-neuron (pre -> post) REIMPLEMENTATION of the relevant DACx.cpp forward
## dynamics, run in plain R at full time resolution, so that a REAL,
## simulated-trace-based Delta g(Delta t) STDP curve can be checked --
## replacing hebbian_check.R's approach of sweeping FIXED, hand-picked scalar
## eta/Hcab values (which produced a magnitude-grows-with-lag artifact, the
## opposite of biological STDP; see that script's closing assessment).
##
## MOTIVATION (from the handwritten sketch, "STDP_model.jpeg", 2026-09-11)
## -------------------------------------------------------------------------
## The sketch's claim is that spike-timing-dependent potentiation falls out
## of the model's own energy accounting (I_total / dHdv / spike_cost, already
## implemented in DACx.cpp lines ~3132-3155) WITHOUT any extra assumption,
## provided you look at the REAL, moment-to-moment trajectory of I_total(t)
## and the synapse's own contribution I_eff_ij(t) around an actual spike --
## not a long-run-average fixed value. Specifically: a presynaptic EPSC that
## arrives close enough to the post-synaptic threshold-crossing does
## productive depolarizing work that is never "wasted" by repolarization
## cutting it off early, and that work substitutes for -- i.e. lowers -- the
## net energy the post-synaptic cell has to spend reaching/paying for its own
## spike. That net saving (a real eta < 0 in Baum-Eagon terms) should be
## largest for some short, causal pre-before-post lag and fall off as the
## lag grows (in either direction) -- the hallmark STDP shape neither rule in
## hebbian_check.R reproduced.
##
## WHAT'S REIMPLEMENTED HERE (from src/DACx.cpp)
## -------------------------------------------------------------------------
## - I_leak(i)              = g_leak(i) * (v(i) - v_rest(i))                 [line 3018]
## - S_fast / S_slow / S_emit gating traces (onset/active/decay)            [2997-3015]
## - v_syn (saturating eq. 26/28/29) and I_syn_effective                    [3096-3115]
## - dHdv (== "I_total" in the sketch), clamped at dHdv_bound                [3132-3139]
## - spike_cost / max_spike_cost / normalized_spike_cost / v_bound_fraction  [3141-3154]
## - dvdt, T modulation, v_sub update, spike detection, last_spike reset     [3156-3214]
##
## DELIBERATE SIMPLIFICATIONS (for a fast, single-synapse toy; all noted so
## they can be relaxed later if this first pass looks promising)
## -------------------------------------------------------------------------
## 1. SOMA SYNAPSE (L_ij = 0): the synapse sits directly on the post-synaptic
##    soma, so met_atten = 1, d = 0 (v_syn_cable = v2 directly, no cable
##    attenuation toward v_rest), and -- per DACx.cpp's own convention for
##    soma synapses (line 229, "seeded with per_nrn.g_leak") -- G_inf = g_leak
##    of the post-synaptic cell. This also makes post_syn_L_norm = 0, so
##    S_slow decays instantly every non-active step (syn_decay_slow = 0):
##    no distance-stretched supra-additive summation at a soma synapse,
##    exactly as in the real model.
## 2. NO axonal/dendritic conduction delay (pre_syn_lag = post_syn_lag = 0):
##    the presynaptic spike's gating onset appears one simulation step after
##    threshold-crossing (the minimal delay inherent to the discrete update
##    itself -- see the ls_lagged/last_spike bookkeeping in DACx.cpp), with
##    no added travel time. Timing effects studied below are therefore purely
##    the LOCAL electrophysiology/energy-accounting effect the sketch is
##    about, not a confound from wire delay.
## 3. NO calcium/vesicle bursting dynamics: Ca is held at 0 and Vs at 1
##    throughout (never updated), so the temporal-modulation term T reduces
##    to a near-constant ~= 1/tau_fast (computed via the real formula, not
##    hardcoded, so this is trivial to relax later). This removes short-term
##    depression/facilitation and the sub-additive Ta/tAe multi-synapse
##    effects (moot anyway with only one synapse) as confounds.
## 4. tA = Ta = 0 for both cell types: no supra-additive same-site-over-time
##    boost either (irrelevant here since a single, isolated pre-spike never
##    pushes S_slow above 1 anyway; S_excess = 0 throughout).
## None of these simplifications touch the mechanism the sketch is about:
## the saturating driving-force term (eq. 26), the synaptic current it
## produces, and how that current enters the SAME dHdv/spike_cost energy
## accounting that determines the post-synaptic spike itself.
##
## =============================================================================

library(tidyverse)

## --- Cell-type-like parameters (both neurons share these; only neuron 2 -----
## --- receives a synapse) -----------------------------------------------------

v_rest        <- -70.0   # mV
v_threshold   <- -55.0   # mV
v_bound_mult  <- 1.15    # multiplier on |v_rest|
v_bound       <- v_bound_mult * abs(v_rest)          # mV
g_leak        <- 10.0    # nS
tau_fast      <- 5.0     # ms
dHdv_bound_mult <- 1.05
I_spike_mag   <- 1000.0  # pA
dHdv_bound    <- dHdv_bound_mult * I_spike_mag        # fW/mV-equivalent bound on dHdv
v_spike_peak  <- 35.0    # mV
tau_spike_ms  <- 1.0     # ms (spike width / synapse activation window)
spike_height  <- v_spike_peak - v_threshold

## Synapse (neuron 1 -> neuron 2), excitatory, ss->ss-like conductance
g_syn12       <- 0.4     # nS
v_eq12        <- 0.0     # mV (excitatory)
tau_syn_fast  <- 2.0     # ms
tA_post       <- 0.0
Ta_post       <- 0.0
G_inf12       <- g_leak  # soma-synapse convention (DACx.cpp line 229)

## Calcium/theta constants (fixed inert values: Ca == 0, Vs == 1 throughout,
## per simplification #3 above -- kept as real formula terms, not hardcoded,
## so bursting dynamics can be reintroduced later without touching this block)
theta_low     <- 0.1
theta_n_const <- theta_low^4

## --- Core simulator: one pre-synaptic spike, one post-synaptic cell --------

#' Simulate a 2-neuron (pre -> post) pair at fixed dt.
#'
#' @param dt          time step, ms
#' @param n_steps     number of steps to simulate
#' @param I_stim1     numeric vector (length n_steps), pA, stimulus to neuron 1 (pre)
#' @param I_stim2     numeric vector (length n_steps) or scalar, pA, stimulus to neuron 2 (post)
#' @param g_syn       synaptic conductance 1->2, nS (default g_syn12; set to 0 to disable)
#' @return a data.frame with one row per time step
simulate_pair <- function(dt, n_steps, I_stim1, I_stim2, g_syn = g_syn12) {

  I_stim1 <- rep_len(I_stim1, n_steps)
  I_stim2 <- rep_len(I_stim2, n_steps)

  tau_spike_steps <- round(tau_spike_ms / dt) + 1   # matches DACx.cpp line 2936
  tau_onset       <- tau_spike_steps - 1             # matches DACx.cpp line 2941
  syn_decay_fast  <- exp(-dt / tau_syn_fast)

  # State vectors (index 1 = neuron 1/pre, index 2 = neuron 2/post)
  v1 <- v2 <- numeric(n_steps)
  v1[1] <- v2[1] <- v_rest
  spikes1 <- spikes2 <- numeric(n_steps)
  last_spike1 <- last_spike2 <- integer(n_steps)

  S_fast <- S_slow <- S_emit <- numeric(n_steps)  # gating trace, synapse 1->2 only
  I_syn_eff2 <- numeric(n_steps)                   # == I_eff_{2,1}(t) in the sketch's notation
  I_leak1 <- I_leak2 <- numeric(n_steps)
  dHdv1 <- dHdv2 <- numeric(n_steps)               # == I_total(t) in the sketch's notation
  spike_cost2 <- numeric(n_steps)
  v_syn_trace <- numeric(n_steps)                  # local synaptic potential v_syn (eq. 26)

  for (t in 2:n_steps) {

    ## --- gating trace update (DACx.cpp lines 2997-3015) --------------------
    ls_lagged <- last_spike1[t - 1]         # no conduction delay (simplification #2)
    active    <- ls_lagged > 0
    onset     <- ls_lagged == tau_onset
    if (active) {
      S_fast[t] <- min(S_fast[t - 1] + as.numeric(onset), 1.0)
      S_slow[t] <- S_slow[t - 1] + as.numeric(onset)        # instant-decay branch not taken while active
    } else {
      S_fast[t] <- syn_decay_fast * S_fast[t - 1]
      S_slow[t] <- 0.0                                       # soma synapse: syn_decay_slow == 0 (L_norm == 0)
    }
    S_excess  <- max(S_slow[t] - 1.0, 0.0)                    # always 0 here (single, isolated pre-spike)
    S_emit[t] <- S_fast[t] + tA_post * S_excess

    ## --- leak currents (line 3018) ------------------------------------------
    I_leak1[t] <- g_leak * (v1[t - 1] - v_rest)
    I_leak2[t] <- g_leak * (v2[t - 1] - v_rest)

    ## --- synaptic current onto neuron 2 (lines 3096-3115), soma synapse ----
    v_syn_cable   <- v2[t - 1]                     # d = 0: no cable attenuation (simplification #1)
    drive_cable   <- v_syn_cable - v_eq12
    drive_eff     <- g_syn / (g_syn + G_inf12)      # Gij (eq. 29)
    Sij_Gij       <- S_emit[t] * drive_eff
    v_syn         <- v_syn_cable - drive_cable * (1.0 - exp(-Sij_Gij))   # eq. 26/28
    drive         <- v_syn - v_eq12
    v_syn_trace[t] <- v_syn
    I_syn_eff2[t] <- g_syn * S_emit[t] * drive      # met_atten = 1, Tae = 1 (simplifications #1, #3)

    ## --- dHdv ("I_total"), spike-cost energy accounting (lines 3132-3154) --
    dHdv1[t] <- min(I_stim1[t - 1] + I_leak1[t] + spikes1[t - 1] * I_spike_mag, dHdv_bound - .Machine$double.eps)
    dHdv2[t] <- min(I_syn_eff2[t] + I_stim2[t - 1] + I_leak2[t] + spikes2[t - 1] * I_spike_mag, dHdv_bound - .Machine$double.eps)

    dvdt_from_energy <- function(dHdv_t, v_prev) {
      spike_repol_power      <- dHdv_bound * v_prev
      rest_maint_power       <- dHdv_t * v_bound
      spike_cost              <- spike_repol_power - rest_maint_power
      spike_repol_from_rest   <- dHdv_bound * v_bound
      maint_power             <- dHdv_t * v_prev
      max_spike_cost           <- spike_repol_from_rest - maint_power
      normalized_spike_cost   <- spike_cost / max_spike_cost
      v_bound_fraction        <- v_bound * normalized_spike_cost
      list(dvdt = v_bound_fraction - v_prev, spike_cost = spike_cost)
    }

    e1 <- dvdt_from_energy(dHdv1[t], v1[t - 1])
    e2 <- dvdt_from_energy(dHdv2[t], v2[t - 1])
    spike_cost2[t] <- e2$spike_cost

    ## --- T modulation (near-constant here; Ca==0, Vs==1 fixed, simplif. #3) --
    Ca_n  <- (1.0 - 0.0)^4
    T_val <- 1.0 * (Ca_n / (Ca_n + theta_n_const)) / tau_fast

    apply_T <- function(dvdt_raw, spikes_prev, last_spike_prev) {
      if (spikes_prev == 1) {
        dvdt_raw                       # reset step: full magnitude, no T scaling
      } else if (last_spike_prev > 0) {
        0.0                            # held during active spike window
      } else {
        dvdt_raw * T_val * dt
      }
    }

    dv1 <- apply_T(e1$dvdt, spikes1[t - 1], last_spike1[t - 1])
    dv2 <- apply_T(e2$dvdt, spikes2[t - 1], last_spike2[t - 1])

    v1[t] <- min(max(v1[t - 1] + dv1, -v_bound), v_threshold)
    v2[t] <- min(max(v2[t - 1] + dv2, -v_bound), v_threshold)

    spikes1[t] <- as.numeric(v1[t] >= v_threshold)
    spikes2[t] <- as.numeric(v2[t] >= v_threshold)

    last_spike1[t] <- max(last_spike1[t - 1] + spikes1[t] * tau_spike_steps - 1, 0)
    last_spike2[t] <- max(last_spike2[t - 1] + spikes2[t] * tau_spike_steps - 1, 0)
  }

  tibble(
    step        = seq_len(n_steps),
    t           = (seq_len(n_steps) - 1) * dt,
    v1, v2, spikes1, spikes2,
    v1_trace    = v1 + spike_height * spikes1,
    v2_trace    = v2 + spike_height * spikes2,
    S_fast, S_slow, S_emit,
    v_syn       = v_syn_trace,
    I_syn_eff2, I_leak1, I_leak2,
    dHdv1, dHdv2,
    spike_cost2
  )
}

## --- Real, trace-based Hebbian update (revised-paper eqs. 36-42, corrected) -
## Unlike hebbian_check.R (which swept eta and Hcab as FREE scalars), every
## quantity below is read off the ACTUAL simulated trace:
##   eta_ij(t)       = I_syn_eff2(t)   -- this channel's own, real, moment-to-
##                                        moment contribution to dHdv2 (there is
##                                        only one channel onto neuron 2 here,
##                                        so this is exact, not an approximation)
##   I_total_i(t)    = dHdv2(t)        -- the real (clamped) total current
##   S_ij(t)         = S_emit(t)       -- the real gating trace
##   v_cab(t)        = v2(t-1)          -- the real cable/soma potential (d=0)
## g_pre_ij (eq. 39) and v_pre_ij (eq. 40) use the CURRENT conductance g_syn,
## held fixed within one exposure (this computes the instantaneous dg/dt at
## each moment, not yet integrating g's own feedback into the trace).
## exp(Lki) == 1 throughout: soma synapse, no cable attenuation (simplif. #1).
dgdt_trace <- function(df, g_syn = g_syn12, Ginf = G_inf12, v_eq = v_eq12, S_floor = 1e-6) {
  S_safe <- pmax(df$S_emit, S_floor)
  g_pre <- (S_safe * g_syn + Ginf)^2 / Ginf                      # eq. 39 (constant while g fixed)
  v_pre  <- S_safe * (v_eq - df$v_syn)                   # eq. 40 (uses local v_syn, i.e. v_cab
                                                          #  at retrieval; matches the script's
                                                          #  v_syn_cable == v2(t-1) here since d=0)
  dv_post_dt <- df$I_syn_eff2 / df$dHdv2                  # eq. 41-42, exp(Lki) == 1
  dgdt <- (g_pre / v_pre) * dv_post_dt
  # Zero out steps with no real gating at all (S_emit numerically 0): no synaptic
  # drive at that instant means no real learning signal from this channel, by
  # construction (I_syn_eff2 == 0 there too), so this just avoids 0/S_floor noise.
  dgdt[df$S_emit <= 0 & df$I_syn_eff2 == 0] <- 0
  dgdt
}

## =============================================================================
## PROTOCOL: single pre-spike -> already-spiking post-neuron, vary the offset
## =============================================================================
## Neuron 2 is given a tonic depolarizing bias strong enough that it spikes
## on its own within the simulated window (autonomously, g_syn = 0) at a
## reference time t_post0. A single, calibrated presynaptic pulse (forcing
## exactly one spike in neuron 1) is then added through a synapse, at a range
## of onset times, so the ACTUAL causal offset Delta t = t_post - t_pre
## (both read off the real simulated spike trains, not assumed) sweeps from
## "pre well before post" down through near-coincidence.

dt        <- 0.02   # ms
n_steps   <- as.integer(150 / dt)
I2_bias   <- -160    # pA, tonic bias -> neuron 2 spikes alone at t_post0 ~= 39.5 ms
g_test    <- 0.4     # nS, ss->ss-like synaptic conductance

pulse_current <- function(dt, n_steps, t_on, dur, amp) {
  I <- numeric(n_steps)
  tt <- (seq_len(n_steps) - 1) * dt
  I[tt >= t_on & tt < (t_on + dur)] <- amp
  I
}

# Baseline: cost of an unaided spike (bias only, synapse silent)
sim_baseline    <- simulate_pair(dt, n_steps, I_stim1 = 0, I_stim2 = I2_bias, g_syn = 0)
t_post_baseline <- sim_baseline$t[which(sim_baseline$spikes2 == 1)[1]]
cost_baseline   <- sim_baseline$spike_cost2[which(sim_baseline$spikes2 == 1)[1] + 1]

run_trial <- function(t_on, g_syn = g_test, I2 = I2_bias, amp = -800, dur = 2) {
  I1  <- pulse_current(dt, n_steps, t_on = t_on, dur = dur, amp = amp)
  sim <- simulate_pair(dt, n_steps, I_stim1 = I1, I_stim2 = I2, g_syn = g_syn)
  t_pre  <- sim$t[sim$spikes1 == 1][1]
  t_post <- sim$t[sim$spikes2 == 1][1]
  dgdt   <- dgdt_trace(sim, g_syn = g_syn)
  tibble(
    t_on, t_pre,
    t_post          = ifelse(length(t_post) == 0, NA, t_post),
    delta_g         = sum(dgdt * dt, na.rm = TRUE),
    spike_cost_paid = if (length(which(sim$spikes2 == 1)) == 0) NA_real_
                       else sim$spike_cost2[which(sim$spikes2 == 1)[1] + 1]
  )
}

t_on_grid <- seq(2, 40, by = 0.5)
stdp_sweep <- map(t_on_grid, run_trial) |>
  list_rbind() |>
  mutate(
    delta_t          = t_post - t_pre,
    cost_savings_mag = abs(cost_baseline) - abs(spike_cost_paid)  # > 0 = cheaper than baseline
  )

## --- Plot A: representative traces (mirrors the sketch's right-hand panel) -

sim_far   <- run_and_return_trace <- function(t_on) {
  I1 <- pulse_current(dt, n_steps, t_on = t_on, dur = 2, amp = -800)
  sim <- simulate_pair(dt, n_steps, I_stim1 = I1, I_stim2 = I2_bias, g_syn = g_test)
  sim$dgdt <- dgdt_trace(sim, g_syn = g_test)
  sim
}
sim_far   <- run_and_return_trace(8)    # Delta t ~= 29 ms (pre well before post)
sim_close <- run_and_return_trace(37)   # Delta t ~= 0.6 ms (pre just before post)

align_df <- bind_rows(
  sim_far   |> mutate(trial = "far (\u0394t \u2248 29 ms)",   t_pre_i = sim_far$t[which(sim_far$spikes1 == 1)[1]]),
  sim_close |> mutate(trial = "close (\u0394t \u2248 0.6 ms)", t_pre_i = sim_close$t[which(sim_close$spikes1 == 1)[1]])
) |>
  mutate(t_rel = t - t_pre_i) |>
  filter(t_rel > -3, t_rel < 35)

p_v <- ggplot(align_df, aes(t_rel, v2_trace, color = trial)) +
  geom_line() +
  coord_cartesian(ylim = c(-82, -50)) +
  labs(y = "v2 (mV)", x = NULL, title = "Postsynaptic voltage (aligned to pre-spike)", color = NULL)

p_total <- ggplot(align_df, aes(t_rel, dHdv2, color = trial)) +
  geom_hline(yintercept = 0, color = "grey60", linewidth = 0.3) +
  geom_line() +
  coord_cartesian(ylim = c(-300, 50)) +
  labs(y = "I_total (dHdv2, pA)\n[spike current off-scale, clipped]", x = NULL,
       title = "I_total(t)  (\"I_total\" in the sketch)", color = NULL)

p_eff <- ggplot(align_df, aes(t_rel, I_syn_eff2, color = trial)) +
  geom_hline(yintercept = 0, color = "grey60", linewidth = 0.3) +
  geom_line() +
  labs(y = "I_eff_21 (pA)", x = "time since presynaptic spike (ms)",
       title = "I_eff_ij(t)  (\"I_eff_ij\" in the sketch)", color = NULL)

library(patchwork)
p_v / p_total / p_eff

## --- Plot B: the two candidate STDP curves, same x-axis (real Delta t) -----

p_cost <- ggplot(stdp_sweep, aes(delta_t, cost_savings_mag)) +
  geom_hline(yintercept = 0, color = "grey50", linetype = "dashed") +
  geom_line(linewidth = 0.9) + geom_point(size = 0.8) +
  labs(x = "\u0394t = t_post \u2212 t_pre (ms, real simulated spike times)",
       y = "|spike_cost| saved vs.\nunaided baseline (fW)",
       title = "(1) Real energetic savings at the post-synaptic spike",
       subtitle = "Directly tests the sketch's literal claim -- uses spike_cost2 only, not the derived Hebbian rule")

p_dg <- ggplot(stdp_sweep, aes(delta_t, delta_g)) +
  geom_hline(yintercept = 0, color = "grey50", linetype = "dashed") +
  geom_line(linewidth = 0.9, color = "firebrick") + geom_point(size = 0.8, color = "firebrick") +
  labs(x = "\u0394t = t_post \u2212 t_pre (ms)",
       y = "integrated \u0394g_ij (nS, whole run)",
       title = "(2) The REAL, trace-based Hebbian update (revised-paper eqs. 36-42)",
       subtitle = "dg/dt integrated using SIMULATED S_emit(t), v_syn(t), I_eff(t), I_total(t) -- no fixed eta/Hcab")

p_cost / p_dg

## Run this script interactively and print/inspect:
##   (p_v / p_total / p_eff)               -- sketch-style trace comparison
##   (p_cost / p_dg)                        -- the two candidate STDP curves

## =============================================================================
## FINDINGS (Sep 11, 2026 session)
## =============================================================================
## (1) THE SKETCH'S LITERAL ENERGETIC CLAIM HOLDS in this toy simulation.
##     p_cost shows a genuine, causal, decaying-with-lag STDP shape that
##     NEITHER rule in hebbian_check.R produced: the post-synaptic spike is
##     measurably CHEAPER (smaller |spike_cost2|) when the pre-synaptic spike
##     lands a short, positive Delta t before it (peak saving right around
##     Delta t ~ 0.1-1 ms, ~1700 fW here), decaying to ~0 saving by
##     Delta t ~ 8-10 ms, and exactly ZERO saving for Delta t < 0 (a pre-spike
##     that arrives only AFTER the post-neuron already spiked cannot possibly
##     have helped it -- correct causality, for free, from the simulation).
##     This is a real confirmation of the mechanism in the sketch's right-hand
##     panel: a well-timed EPSC does productive depolarizing work that
##     substitutes for part of what the cell would otherwise have had to pay
##     to reach threshold on its own.
##
## (2) BUT the DERIVED Hebbian rule (eqs. 36-42, corrected, exactly as used in
##     hebbian_check.R's "NEW rule") does NOT inherit this shape when driven
##     by the same real traces. p_dg instead reproduces hebbian_check.R's
##     flagged artifact: magnitude GROWS (not decays) as the pre-spike moves
##     further into the past, peaking around Delta t ~ 28-29 ms here before
##     falling off only near the edge of the simulated window.
##     Diagnosed cause (see split_contrib() exploration in the session
##     transcript): almost all of dg/dt's integral for LARGE Delta t accrues
##     BEFORE the post-spike, while the EPSC decays through an otherwise-quiet
##     I_total regime (dHdv2 staying small and steadily negative for many ms).
##     For SMALL Delta t, the EPSC's decay tail instead overlaps the post-
##     spike/reset event itself, where I_total swings to a completely
##     different, clamped, oppositely-signed regime (dominated by
##     spikes2*I_spike_mag ~ +1000 pA) -- so the dv_post_ij/dt = eta_ij/I_total
##     term used at that instant is dividing by a value that has little to do
##     with the "quiet" energetics the rest of the curve is sampling, and the
##     accumulated integral partially cancels instead of adding constructively.
##     In short: the SAME simulated data shows a real STDP effect in the
##     model's own spike-cost accounting (1), but the specific chain-rule
##     construction of eta_ij = I_syn_eff2(t), I_total_i = dHdv2(t) at each
##     instant (2) does not transmit that effect into dg/dt -- because the
##     denominator I_total(t) is not a "slowly varying Hcab" near a spike (the
##     premise hebbian_check.R's fixed-Hcab sweep implicitly assumed), it is
##     itself violently timing-dependent in a way that fights the numerator.
##
## OPEN QUESTION THIS RAISES (not resolved here): eq. 38's dv_post_ij/dt =
## eta_ij / (exp(Lki)*I_total,i) was derived assuming I_total,i is the right
## normalizer at every instant. Finding (2) suggests that instant-by-instant
## ratio is the wrong thing to integrate through a spike; the sketch's own
## intuition (1) is about a DIFFERENT quantity -- the NET spike-cost saved by
## the whole episode, evaluated once (at/after the spike), not a running
## eta/I_total ratio sampled continuously through the spike transient. Worth
## deciding, before any C++ implementation, whether the update rule should be
## re-derived to use a windowed/one-shot spike_cost comparison (matching (1))
## rather than literal instant-by-instant integration of eq. 38 (which gives
## (2)). A useful side observation from this same exercise: because eta_ij and
## v_pre_ij are BOTH tied to the real S_ij(t) trace in the actual model (not
## independent free parameters as in hebbian_check.R), the low-S divergence
## that script flagged as a shared flaw of both old/new rules is much less
## severe here -- S cancels approximately between I_syn_eff2 (proportional to
## S) and v_pre_ij (also proportional to S), leaving dg/dt's magnitude set
## mostly by drive/I_total rather than blowing up as S -> 0.
## =============================================================================
