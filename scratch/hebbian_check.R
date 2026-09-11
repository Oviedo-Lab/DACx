## =============================================================================
## scratch/hebbian_check.R
## Visualizing the Hebbian learning rule in DACx (paper sec. 2.1.6) -- REVISED
##
## PURPOSE
## -------
## The DACx paper derives a local Hebbian-style learning rule from the Baum-
## Eagon inequality (eq. 3) and the synaptic electrophysiology model (sec.
## 2.1.5). As of Sep 2026, this rule has NOT been implemented in the C++
## source (src/DACx.cpp) -- only the forward (non-learning) synaptic dynamics
## are implemented there. This script checks whether the rule, as written, is
## consistent with "neurons that fire together, wire together" before ever
## coding it.
##
## THIS VERSION (Sep 11, 2026, second pass) tracks the REVISED paper
## (DACxMini_Hebb_revised), which changed equation 26 (the local synaptic
## potential v_ij) from a linear sum of "fast" and "slow" potentials to a
## single SATURATING exponential term. That change was implemented in the
## forward dynamics at src/DACx.cpp line ~3101 (see RELATED CHANGES below)
## and, because eq. 26 feeds directly into the Hebbian derivation (sec.
## 2.1.6), it also changes the local synaptic update rule itself (eq. 44 in
## the original paper, eq. 38 in the revised paper).
##
## SECOND-PASS CORRECTION: while checking the first pass of this analysis, I
## found and reported a chain-rule slip in the paper's Appendix B.1 (eq. 53),
## which had propagated into eq. 36 and, through it, eq. 38's g_pre_ij/v_pre_ij
## form. You confirmed and fixed eq. 36 in the paper, and confirmed that
## appendix eqs. 59-60 are stale hold-over text to be ignored entirely (they
## were never derived from the current eq. 36/37 and should not be used as a
## source for anything). This version of the script implements the update
## rule using the now-CORRECTED eq. 36, and the resulting formula is
## different -- in a couple of qualitatively important ways -- from what the
## previous (buggy) version of this script reported. See "WHAT CHANGED
## BETWEEN THE FIRST AND SECOND PASS" below.
##
## THIRD-PASS CORRECTION: the original paper explicitly restricts eta to be
## POSITIVE (main text: "Suppose dHi/dt = -eta for eta > 0"), so its sign
## convention (eta = -dHi/dt) bakes in the assumption that this synaptic
## channel's marginal contribution to power is always decreasing. The revised
## paper drops this restriction (main text: "Suppose dHi/dt = eta", with no
## positivity constraint) -- eta = dHi/dt can be either sign. This is
## physically sensible: Baum-Eagon (eq. 3) only requires a neuron's TOTAL
## power to be non-increasing; the marginal contribution attributed to one
## synaptic pathway in isolation (the specific chain vi -> vij -> gij used to
## derive eqs. 34-38) need not individually be non-increasing, so long as the
## sum over all channels still is. Plot 1's eta axis was widened from
## [0.5, 100] to [-100, 100] to reflect this, and the result is genuinely
## informative -- see "WHAT ALLOWING NEGATIVE ETA REVEALS" below.
##
## HOW TO RUN
## ----------
## 1. library(DACx)   -- or install with devtools::install()
## 2. source("scratch/hebbian_check.R")
## Plots are printed to the active device. No network simulation is needed.
## The G_inf and g_syn values used here were derived offline from the EI-loops
## vignette network (see "Biologically realistic parameters" section below).
##
## BACKGROUND: WHAT CHANGED IN EQ. 26
## -----------------------------------
## OLD (original paper, eq. 26):
##   v_ij = v_cab_ij - (v_fast_ij + v_slow_ij)
##   v_fast_ij = S_fast_ij       * G_ij * (v_cab_ij - v_eq_ij)     (linear in S_fast)
##   v_slow_ij = phi*max(S_slow_ij-1,0) * G_ij * (v_cab_ij - v_eq_ij)  (linear in S_slow)
## This is UNBOUNDED: for large enough S_ij = S_fast_ij + phi*max(S_slow_ij-1,0),
## v_ij can overshoot past v_eq_ij (nonsensical for a driving-force term).
##
## NEW (revised paper, eqs. 26/28/29):
##   v_ij = v_cab_ij + v_input_ij
##   v_input_ij = (v_eq_ij - v_cab_ij) * (1 - exp(-S_ij * G_ij))    (saturating)
##   G_ij = g_ij / (g_ij + G_inf_ki(j))                              (unchanged, eq. 29)
## As S_ij*G_ij -> Inf, v_ij -> v_eq_ij (saturates, never overshoots). This is
## a genuine improvement to the forward dynamics: it fixes a physically
## nonsensical overshoot that was possible under the old linear model whenever
## a synapse fired repeatedly (S_slow uncapped) at a proximal site (G_ij large).
##
## HOW THIS CHANGES THE HEBBIAN UPDATE RULE (sec. 2.1.6)
## --------------------------------------------------------
## Both papers derive the update rule for spike-dependent conductance g_ij by
## expanding the Baum-Eagon inequality:
##   (dH_i/dv_i)(dv_i/dv_ij)(dv_ij/dg_ij)(dg_ij/dt) <= 0                 (eq. 34)
## and, assuming dH_i/dt = eta for a non-steady-state cell, solving for dg_ij/dt.
##
## OLD RULE (original paper, eqs. 37-44), derived from the LINEAR eq. 26:
##   dg_ij/dvij = -(g+Ginf)^2 / (S*Ginf*(vcab-veq))              (eq. 37, no exponential)
##   dvij/dvi   = exp(-Lki)*(1 - S*Gij)                          (eq. 38, linear in S*Gij)
##   => dg_ij/dt = eta/Hcab_ij * ( G2(g)*H(S) - O(g) )           (eq. 44, AS PRINTED)
##   where G2(g) = Ginf + 2g + g^2/Ginf   ("hyperbolic slope")
##         H(S)  = -1/S                   ("hyperbolic decay")
##         O(g)  = g + g^2/Ginf           ("hyperbolic offset")
##   and Hcab_ij = (vcab_ij - veq_ij) * exp(Lki(j)) * I_syn,i.
##   NOTE: an earlier version of this script found that the bracket needed to
##   be -( G2(g)*H(S) + O(g) ) (both the overall sign AND the sign in front of
##   O flipped) to produce the sign-flip-at-S* behavior documented below; the
##   code below (dg_dt_old) implements that corrected form, not the bracket
##   exactly as printed in the paper (which never changes sign for fixed-sign
##   Hcab, since G2*H(S) and -O(g) are then both negative).
##
## NEW RULE (revised paper, eqs. 36-42, NOW CORRECTED), derived from the
## SATURATING eq. 26:
##   dg_ij/dvij = (g+Ginf)^2/(Ginf*(veq-vcab)) * exp(S*Gij)/S      (eq. 36, corrected:
##                                                                   divide by S, not S*Gij)
##   dvij/dvi   = exp(-(Lki + S*Gij))                              (eq. 37, unchanged, verified
##                                                                   against a numerical
##                                                                   derivative last session)
##   => dg_ij/dt = (g_pre_ij / v_pre_ij) * dv_post_ij/dt            (eq. 38)
##      g_pre_ij = (g+Ginf)^2 / Ginf                                (eq. 39, unchanged)
##      v_pre_ij = S*(veq - vcab)                                   (eq. 40, corrected: no Gij)
##      dv_post_ij/dt = eta / (exp(Lki) * I_total,i)                (eqs. 41-42, unchanged)
##
## Substituting, the exp(S*Gij) and exp(-S*Gij) factors still cancel exactly,
## and writing Hcab_ij = (vcab_ij - veq_ij) * exp(Lki(j)) * I_total,i (NOTE:
## I_total,i, not I_syn,i -- see CAVEAT below), this now collapses to:
##
##   dg_ij/dt = (eta / Hcab_ij) * G2(g) * H(S)                    ***NEW RULE***
##            = -(eta / Hcab_ij) * (g + Ginf)^2 / (Ginf * S)        [equivalent form]
##
## i.e. the new rule is now EXACTLY the old rule's G2(g)*H(S) term, with the
## offset O(g) simply absent (no cube/g->0 divergence, unlike the leftover-bug
## version of this script). Verified numerically to agree with eq. 38 computed
## literally from the corrected eqs. 39-42, and with a plain numerical
## derivative of eq. 26/28+27 -- see the R console transcript from this
## session for the checks (not reproduced as inline code here, to keep this
## script focused on the final, agreed-upon rule).
##
## WHAT CHANGED BETWEEN THE FIRST AND SECOND PASS
## ---------------------------------------------------
## The bug fix (dividing eq. 36 by S instead of S*Gij) does two things:
## 1. REMOVES the small-g divergence. In the buggy version, G2(g)/(S*Gij) blew
##    up as g -> 0 (since Gij -> g/Ginf -> 0). The corrected G2(g)*H(S) does
##    NOT have this problem: G2(g) -> Ginf (finite) as g -> 0, exactly like
##    the old rule. This concern from the previous pass is now resolved.
## 2. FLIPS the sign of the dominant (low-S) term relative to what the buggy
##    version reported. The buggy version had dg/dt = +(eta/Hcab)*G2/(S*Gij)
##    (positive for Hcab>0), i.e. strong LTP as S -> 0. The corrected version
##    has dg/dt = (eta/Hcab)*G2*H(S) = -(eta/Hcab)*G2/S (negative for Hcab>0),
##    i.e. strong LTD as S -> 0. This is the OPPOSITE conclusion from the
##    buggy version and is worth flagging clearly: for Hcab>0 (excitatory,
##    postsynaptic power falling), the corrected new rule STRONGLY DEPRESSES
##    weakly/rarely-driven synapses, where the buggy version had appeared to
##    strongly potentiate them.
##
## WHAT'S DIFFERENT ABOUT THE (CORRECTED) NEW RULE, vs. the OLD RULE
## --------------------------------------------------------------------
## 1. NO MORE OFFSET TERM O(g). The old rule is G2(g)*H(S) - O(g) [as coded,
##    with the sign correction: -(G2*H(S)+O)]: a combination of two terms.
##    The new rule is the single term G2(g)*H(S), with NO added/subtracted
##    offset. Two direct consequences:
##      a. sign(dg/dt) = sign(eta)*sign(Hcab)*sign(H(S)) = -sign(eta)*sign(Hcab)
##         ALWAYS (since H(S)=-1/S<0 for all S>0). The presynaptic-drive-
##         dependent LTP/LTD sign flip that the old rule had at S* = G2(g)/O(g)
##         is gone: the new rule's sign never depends on S or g.
##      b. As S -> Infinity (strong, sustained presynaptic drive), the new
##         rule's magnitude -> 0 (since H(S) -> 0). The OLD rule instead
##         approaches a nonzero floor, -(eta/Hcab)*O(g), as S -> Infinity.
##         So: under the new rule, a synapse driven at very high, sustained S
##         essentially stops changing; under the old rule, it keeps
##         approaching a fixed nonzero LTD rate (for Hcab>0).
## 2. NO NEW SINGULARITY AT g -> 0 (this concern from the buggy version is
##    resolved by the fix; see above). Both old and new rules are finite as
##    g -> 0. Both still diverge as S -> 0 (the 1/S term is common to both;
##    this is not new to either rule).
## 3. Hcab_ij's definition changed from using I_syn,i (original paper, eq. 40)
##    to I_total,i (revised paper, eq. 42, inherited directly from the g_pre/
##    v_pre derivation). I_total,i = I_syn,i + I_stim,i + I_leak,i + I_spike,i
##    (eq. 8) is dominated by the spike current at/around a postsynaptic
##    spike, so Hcab_ij's sign and magnitude near a spike may look very
##    different than when built from I_syn,i alone. This is flagged as an
##    open question, not resolved here -- for the plots below we hold Hcab
##    fixed at representative values as before, so this substitution does not
##    change the qualitative comparison, but it should be revisited before any
##    C++ implementation.
##
## WHAT ALLOWING NEGATIVE ETA REVEALS (Plot 1)
## ----------------------------------------------
## Widening eta to [-100, 100] exposes a real structural difference between
## the two rules that the eta>0-only view couldn't show, because sign(eta)
## is one of the two factors that can flip dg/dt's sign for the OLD rule, but
## the ONLY factor that can flip it for the NEW rule:
##   OLD: dg/dt = -(eta/Hcab)*(G2(g)*H(S) + O(g)). Flipping eta's sign flips
##        the overall sign, AND crossing S* independently flips the sign of
##        the bracket. These two switches are independent, so the (S, eta)
##        plane splits into FOUR alternating sign regions (confirmed
##        numerically: at Hcab=+600, (eta,S)=(90,0.05) -> +5.83 (LTP),
##        (90,2) -> -0.069 (LTD), (-90,0.05) -> -5.83 (LTD), (-90,2) -> +0.069
##        (LTP)). This is visible in Plot 1's left panel as a "+"-shaped
##        contour (a horizontal line at eta=0 AND a vertical line at S*).
##   NEW: dg/dt = (eta/Hcab)*G2(g)*H(S). Since G2(g)>0 and H(S)<0 always, the
##        sign is ALWAYS exactly -sign(eta)*sign(Hcab), with NO S-dependence
##        (confirmed: (90,0.05) -> -6.05 (LTD), (90,2) -> -0.151 (LTD),
##        (-90,0.05) -> +6.05 (LTP), (-90,2) -> +0.151 (LTP)). Plot 1's right
##        panel shows only ONE contour line (horizontal, at eta=0) -- no
##        vertical line at any S.
## This gives the NEW rule a clean, single-cause interpretation for Hcab>0
## (excitatory): if this synapse's own marginal contribution to somatic power
## is currently making Hi increase (eta>0 -- working against the network's
## overall Baum-Eagon energy minimization), the rule weakens it (LTD); if its
## marginal contribution is instead helping Hi decrease (eta<0), the rule
## strengthens it (LTP). That's a coherent "reward energy-decreasing synaptic
## contributions" story with no confound.
## The OLD rule does NOT have this clean interpretation: a synapse whose
## marginal contribution is INCREASING power (eta>0, the same "bad" signal as
## above) can still get POTENTIATED under the old rule, provided S happens to
## be below S*. The direction of learning there depends on an unrelated
## quantity (how strongly the synapse happens to be driven right now), not
## just on whether the channel is helping or hurting the energy-minimization
## goal. On reflection, I think this is a real point in the NEW rule's favor,
## independent of the earlier (algebraic) discussion of the eq. 36 fix.
##
## FOURTH-PASS: Plots 2-4 widened to include negative eta too (previously
## positive-only). Since both rules are exactly LINEAR in eta (eta appears
## only as an overall multiplicative factor -- it never appears inside G2(g),
## H(S), or O(g)), widening eta's sign does NOT reveal any new curve shapes:
## every curve for eta<0 is exactly the eta>0 curve reflected through zero.
## What it DOES do is make the sign-flip structure found in Plot 1 visible in
## these other views too (e.g. Plot 4 now shows the old rule crossing zero
## along g for negative eta where it didn't for positive eta, etc.) -- see
## each plot's own comments below for specifics.
##
## OTHER PARAMETERS EXPLORED FOR "WIDENING" BEHAVIOR
## ------------------------------------------------------
## eta and Hcab both enter the rules purely as an overall multiplicative
## factor (eta/Hcab); varying their MAGNITUDE only rescales the curves, and
## varying Hcab's SIGN is equivalent to flipping eta's sign (already covered
## above). Neither is a source of new qualitative behavior on its own.
## g and Ginf, by contrast, enter NONLINEARLY (through G2(g,Ginf) and, for the
## old rule, O(g,Ginf) too) and both vary substantially across real cell types
## in the EI-loops network (g_syn: 0.4 nS ss->ss vs 4.0 nS PV->ss, a 10x
## range; G_inf: 0.03-0.28 nS, another ~10x range). Since the old rule's
## sign-flip threshold S* = G2(g,Ginf)/O(g,Ginf) is a function of exactly
## these two parameters, it seemed worth checking how much S* actually moves
## across this realistic range -- see Plot 5 (new).
##
## SIGN OF Hcab FOR AN EXCITATORY SYNAPSE (outward-positive convention)
## ---------------------------------------------------------------------
## Retained from the previous version of this script for comparability:
## v_cab - v_eq: for excitatory (v_eq ~ 0 mV), v_cab ~ v_rest ~ -70 mV => ~-60 mV
## exp(L):       always positive
## => Hcab ~ (-60) * 1 * I_i ~ +600 fW for a modestly active postsynaptic cell
##    with net inward (negative) current I_i ~ -10 pA (regardless of whether
##    I_i is I_syn,i or I_total,i, the SIGN is the same in this regime).
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
## WHAT EACH PLOT SHOWS
## --------------------
## Plot 1 (heatmap): dg/dt over the (S_ij, eta) plane, OLD rule vs NEW rule,
##   side by side, for Hcab = +600 (excitatory). The old rule's zero-contour
##   (sign flip at S*) is visible; the new rule has no zero contour at all in
##   this range (sign is fixed, and now NEGATIVE throughout, opposite of the
##   old rule's small-S region).
##
## Plot 2 (line plot): dg/dt vs S at fixed eta values (now +-5, +-50), OLD vs
##   NEW. For eta>0: old rule crosses zero at S*, new rule stays negative and
##   decays to 0. For eta<0: every curve is exactly reflected through zero
##   (linearity in eta), so the old rule now crosses zero from below, and the
##   new rule is positive throughout instead of negative.
##
## Plot 3 (instantaneous probe): dg/dt at the moment pre- and post-synaptic
##   pulses first overlap, OLD vs NEW rule, as a function of delta_t = t_post -
##   t_pre, now shown for BOTH eta_peak = +50 and eta_peak = -50 (facet
##   columns). Isolates timing-dependence from the 1/S blow-up (as in the
##   previous version of this script). Both rules still show the same
##   anti-Hebbian-shaped timing dependence (magnitude growing, not decaying,
##   as the pre spike gets further in the past) for either sign of eta_peak;
##   flipping eta_peak's sign just flips which direction (LTP/LTD) that
##   growth points in, for both rules, exactly as linearity predicts.
##
## Plot 4: dg/dt vs g at fixed S, now for BOTH eta=+20 and eta=-20 (facet
##   columns), and with g extended out to 4.5 nS to cover the PV->ss regime
##   (g_syn=4.0 nS) as well as ss->ss (g_syn=0.4 nS, still marked). Confirms
##   the new rule is bounded as g -> 0 for either sign of eta, just like the
##   old rule (the divergence seen in the buggy version of this script is
##   gone) -- and shows both rules keep growing in magnitude (roughly
##   quadratically) out to the PV->ss conductance range, so neither rule is
##   "safe" at large g either.
##
## Plot 5 (NEW): the old rule's sign-flip threshold S* = G2(g,Ginf)/O(g,Ginf)
##   plotted as a heatmap over the realistic (g, Ginf) range spanning ss->ss
##   through PV->ss synapses and near-soma through distal dendritic sites.
##   The new rule has no such threshold (its sign never depends on g or Ginf
##   at all), so there's nothing analogous to plot for it -- the point of this
##   plot is precisely to show how much this extra, S-and-now-also-(g,Ginf)-
##   dependent switch moves around in the old rule, underscoring how much
##   simpler the new rule's sign structure is by comparison.
##
## RELATED CHANGES MADE IN THIS SESSION (Sep 11, 2026)
## ------------------------------------------------------
## - src/DACx.cpp lines ~3099-3109: replaced the linear v_syn_fast/v_syn_slow
##   sum with the saturating exponential form of eq. 26 (revised):
##     Sij_Gij = S_emit(i,j) * drive_eff;            // S_ij * G_ij
##     v_syn   = v_syn_cable - drive_cable_j * (1 - exp(-Sij_Gij));
##   S_fast, S_slow, S_excess, and S_emit themselves are UNCHANGED (eq. 25's
##   S_ij modulation term and the dendrite_states/Phi superadditivity pathway
##   do not depend on eq. 26 and were not touched).
## - Package reinstalled (devtools::install) after the C++ change.
## - Eq. 36 (and the derived eq. 38 form) corrected in the paper (chain-rule
##   fix: divide by S_ij, not S_ij*G_ij). Appendix eqs. 59-60 confirmed to be
##   stale hold-over text and are to be ignored entirely -- not used anywhere
##   in this script.
## - This script rewritten (second pass) to implement the corrected rule.
## =============================================================================

library(tidyverse)

## --- Core equations: OLD rule (original paper, sec. 2.1.6) ------------------

G2  <- function(g, Ginf) Ginf + 2 * g + g^2 / Ginf       # eq. 41 (both papers)
Hf  <- function(S)       -1 / S                            # eq. 42 (original) "hyperbolic decay"
Oij <- function(g, Ginf)  g + g^2 / Ginf                  # eq. 43 (original) "hyperbolic offset"

dg_dt_old <- function(eta, Hcab, g, Ginf, S) {
  -(eta / Hcab) * (G2(g, Ginf) * Hf(S) + Oij(g, Ginf))    # eq. 44 (corrected sign, per prior
                                                            # version of this script -- NOT the
                                                            # bracket exactly as printed in the
                                                            # paper, which does not produce an
                                                            # S*-threshold sign flip at all)
}

## --- Core equations: NEW rule (revised paper, sec. 2.1.6, eq. 36 corrected) -
## dg/dt = (eta/Hcab) * G2(g) * H(S)  ==  -(eta/Hcab) * (g+Ginf)^2 / (Ginf*S)
## Both forms implemented for cross-checking; no Gij dependence remains after
## the eq. 36 fix (Gij's exp(S*Gij) and exp(-S*Gij) factors cancel exactly,
## and the erroneous extra 1/Gij is gone).

dg_dt_new <- function(eta, Hcab, g, Ginf, S) {
  (eta / Hcab) * G2(g, Ginf) * Hf(S)
}

dg_dt_new_alt <- function(eta, Hcab, g, Ginf, S) {
  -(eta / Hcab) * (g + Ginf)^2 / (Ginf * S)
}

## --- Biologically realistic parameters --------------------------------------

g0    <- 0.4    # nS  (ss->ss g_syn, from EI-loops network cell type)
Ginf0 <- 0.15   # nS  (mid-dendritic G_inf, from eq. 18 + cable params above)

S_star_old <- G2(g0, Ginf0) / Oij(g0, Ginf0)
message(sprintf(
  "OLD rule sign-flip threshold S* = %.2f  (no analogous threshold exists for the NEW rule)",
  S_star_old
))

## --- Sanity check: the two closed forms of the NEW rule agree ---------------

check_grid <- expand.grid(
  eta  = c(0.5, 5, 50),
  Hcab = c(-600, 600),
  g    = c(0.05, 0.1, 1),
  S    = c(0.05, 0.3, 0.8, 2)
) |>
  mutate(
    lhs = dg_dt_new(eta, Hcab, g, Ginf0, S),
    rhs = dg_dt_new_alt(eta, Hcab, g, Ginf0, S),
    ok  = abs(lhs - rhs) < 1e-9 * pmax(1, abs(rhs))
  )
stopifnot(all(check_grid$ok))
message(sprintf(
  "Consistency check passed: G2(g)*H(S) == -(g+Ginf)^2/(Ginf*S) at all %d test points.",
  nrow(check_grid)
))

## --- Plot 1: heatmap of dg/dt over (S, eta), OLD vs NEW, Hcab = +600 --------
## The old rule's zero-contour (sign flip at S*) is visible on the left. The
## new rule (right) has no zero crossing in this range: it is negative
## throughout (LTD), with magnitude set purely by G2(g)*H(S).

grid1 <- expand.grid(
  S    = seq(0.01, 3, length.out = 200),
  eta  = seq(-100, 100, length.out = 200)
) |>
  mutate(
    old = dg_dt_old(eta, 600, g0, Ginf0, S),
    new = dg_dt_new(eta, 600, g0, Ginf0, S)
  ) |>
  pivot_longer(c(old, new), names_to = "rule", values_to = "dgdt") |>
  mutate(rule = factor(rule, levels = c("old", "new"),
                        labels = c("OLD rule (eq. 44, original paper)",
                                   "NEW rule (eq. 38, revised paper, corrected)")))

p1 <- ggplot(grid1, aes(S, eta, fill = dgdt)) +
  geom_raster() +
  geom_contour(aes(z = dgdt), breaks = 0, color = "black", linewidth = 0.5) +
  scale_fill_gradient2(low = "firebrick", mid = "white", high = "steelblue", midpoint = 0) +
  facet_wrap(~rule) +
  labs(
    x        = "S_ij  (presynaptic gating trace; S_emit can exceed 1 via uncapped S_slow)",
    y        = "eta  (fW/ms, proxy for postsynaptic spike-driven power loss)",
    fill     = "dg/dt\n(nS/ms)",
    title    = "OLD vs NEW conductance-change rate over (S, eta), Hcab = +600 fW (excitatory)",
    subtitle = sprintf(
      "OLD: sign flips at BOTH eta=0 and S* = %.2f (4 alternating regions). NEW: sign flips ONLY at eta=0.",
      S_star_old
    )
  )
print(p1)

## --- Plot 2: dg/dt vs S at fixed eta values, OLD vs NEW ---------------------

grid2 <- expand.grid(
  S    = seq(0.01, 3, length.out = 300),
  eta  = c(-50, -5, 5, 50)    # fW/ms; widened to include negative eta
) |>
  mutate(
    old = dg_dt_old(eta, 600, g0, Ginf0, S),
    new = dg_dt_new(eta, 600, g0, Ginf0, S)
  ) |>
  pivot_longer(c(old, new), names_to = "rule", values_to = "dgdt") |>
  mutate(
    rule    = factor(rule, levels = c("old", "new"),
                      labels = c("OLD rule (eq. 44, original paper)",
                                 "NEW rule (eq. 38, revised paper, corrected)")),
    eta_lab = factor(paste0(eta, " fW/ms"), levels = paste0(sort(unique(eta)), " fW/ms"))
  )

p2 <- ggplot(grid2, aes(S, dgdt, color = eta_lab)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
  geom_vline(data = tibble(rule = levels(grid2$rule)[1], xint = S_star_old),
             aes(xintercept = xint), linetype = "dotted", color = "black", linewidth = 0.4) +
  geom_line(linewidth = 0.8) +
  facet_wrap(~rule, scales = "free_y") +
  labs(
    x        = "S_ij (presynaptic gating trace)",
    y        = "dg/dt  (nS/ms)",
    color    = "eta",
    title    = "Conductance change rate vs presynaptic drive, Hcab = +600 fW",
    subtitle = "Dotted line (left panel only) = OLD rule's sign-flip threshold S*"
  )
print(p2)

## --- Plot 3: instantaneous learning-rate-at-coincidence probe, OLD vs NEW ---
## Same construction as before: evaluate dg/dt at the single moment the pre-
## and post-synaptic pulses first overlap, to isolate timing-dependence from
## the 1/S blow-up.
##
## tau_S   = tau_syn_fast = 2 ms (ss->ss, from cell type params)
## tau_eta = 5 ms (phenomenological postsynaptic spike-power-loss decay)
## S_peak  = 1   (one BGT spike increments S_fast by 1, then S_fast is capped at 1)
## eta_peak = 50 fW/ms (near-spike peak power-dissipation rate)

tau_S    <- 2.0    # ms  (tau_syn_fast, ss->ss)
tau_eta  <- 5.0    # ms  (postsynaptic power-loss decay)
S_peak   <- 1.0    # single spike increments S_fast by 1 (then capped at 1 in BGT)
eta_peak <- 50.0   # fW/ms (magnitude; sign now varied too, see eta_sign below)

deltas_fine <- seq(-20, 20, by = 0.25)

rate_at_coincidence <- function(delta_t, rule_fn, eta_sign = 1) {
  if (delta_t >= 0) {
    S_at_post <- S_peak * exp(-delta_t / tau_S)
    rule_fn(eta_sign * eta_peak, 600, g0, Ginf0, S_at_post)
  } else {
    eta_at_pre <- eta_sign * eta_peak * exp(delta_t / tau_eta)   # delta_t < 0
    rule_fn(eta_at_pre, 600, g0, Ginf0, S_peak)
  }
}

stdp_instant <- expand_grid(
    delta_t  = deltas_fine,
    rule_lbl = c("old", "new"),
    eta_sign = c(1, -1)
  ) |>
  rowwise() |>
  mutate(
    dgdt_at_coincidence = rate_at_coincidence(
      delta_t, if (rule_lbl == "old") dg_dt_old else dg_dt_new, eta_sign
    )
  ) |>
  ungroup() |>
  mutate(
    rule = factor(rule_lbl, levels = c("old", "new"),
                  labels = c("OLD rule (eq. 44, original paper)",
                             "NEW rule (eq. 38, revised paper, corrected)")),
    eta_lab = factor(paste0("eta_peak = ", ifelse(eta_sign > 0, "+", "-"), eta_peak, " fW/ms"),
                      levels = paste0("eta_peak = ", c("+", "-"), eta_peak, " fW/ms")),
    side = ifelse(delta_t >= 0, "pre then post  (solid)", "post then pre  (dashed)")
  )

p3 <- ggplot(stdp_instant, aes(delta_t, dgdt_at_coincidence, linetype = side)) +
  geom_hline(yintercept = 0, color = "grey50", linetype = "dashed") +
  geom_vline(xintercept = 0, color = "grey50", linetype = "dashed") +
  geom_line(linewidth = 0.9) +
  facet_grid(rule ~ eta_lab, scales = "free_y") +
  scale_linetype_manual(values = c("pre then post  (solid)"  = "solid",
                                   "post then pre  (dashed)" = "dashed")) +
  labs(
    x        = "post-spike time \u2212 pre-spike time  (ms,  0 = coincident firing)",
    y        = "dg/dt at moment of coincidence  (nS/ms)",
    linetype = NULL,
    title    = "Instantaneous learning rate at coincidence, OLD vs NEW rule (Hcab = +600 fW)",
    subtitle = paste0(
      "Both rules grow in magnitude (not decay) as the pre spike gets further in the past --\n",
      "still anti-Hebbian in the causal-timing sense, for EITHER sign of eta_peak (linearity in\n",
      "eta just flips which direction, LTP/LTD, that growth points in for both rules identically)."
    )
  )
print(p3)

## --- Plot 4: dg/dt vs g at fixed S, OLD vs NEW ------------------------------
## With eq. 36 corrected, the new rule is now bounded as g -> 0, matching the
## old rule's behavior (the divergence seen in the earlier, buggy version of
## this script is gone).

g_PV <- 4.0   # nS  (PV->ss g_syn, from EI-loops network cell type; upper end of realistic range)

grid4 <- expand.grid(
  g        = seq(0.01, 4.5, length.out = 400),
  S        = c(0.3, 1, 2),
  eta_sign = c(1, -1)
) |>
  mutate(
    old = dg_dt_old(eta_sign * 20, 600, g, Ginf0, S),
    new = dg_dt_new(eta_sign * 20, 600, g, Ginf0, S)
  ) |>
  pivot_longer(c(old, new), names_to = "rule", values_to = "dgdt") |>
  mutate(
    rule    = factor(rule, levels = c("old", "new"),
                      labels = c("OLD rule (eq. 44, original paper)",
                                 "NEW rule (eq. 38, revised paper, corrected)")),
    S_lab   = factor(paste0("S = ", S)),
    eta_lab = factor(paste0("eta = ", ifelse(eta_sign > 0, "+", "-"), "20 fW/ms"),
                      levels = paste0("eta = ", c("+", "-"), "20 fW/ms"))
  )

p4 <- ggplot(grid4, aes(g, dgdt, color = S_lab)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
  geom_vline(xintercept = g0,   linetype = "dotted", color = "black", linewidth = 0.4) +
  geom_vline(xintercept = g_PV, linetype = "dotted", color = "grey40", linewidth = 0.4) +
  geom_line(linewidth = 0.8) +
  facet_grid(rule ~ eta_lab, scales = "free_y") +
  labs(
    x        = "g_ij  (spike-dependent conductance, nS)",
    y        = "dg/dt  (nS/ms)",
    color    = NULL,
    title    = "Conductance change rate vs g_ij, Hcab = +600 fW",
    subtitle = "Dotted lines = g0 = 0.4 nS (ss->ss, black) and g_PV = 4.0 nS (PV->ss, grey). Both rules bounded as g -> 0."
  )
print(p4)

## --- Plot 5 (NEW): OLD rule's sign-flip threshold S* over (g, Ginf) ---------
## g and Ginf both enter G2(g,Ginf) and O(g,Ginf) nonlinearly, so -- unlike
## eta and Hcab -- varying them can change more than just an overall scale.
## The old rule's sign-flip threshold, S* = G2(g,Ginf)/O(g,Ginf), is plotted
## across the realistic range spanning ss->ss (g=0.4) through PV->ss (g=4.0)
## conductances, and near-soma through distal-dendrite (Ginf=0.03 to 0.28)
## admittances. The new rule has no such threshold at all (it's sign never
## depends on g or Ginf), so there is nothing analogous to overlay for it.

grid5 <- expand.grid(
  g    = seq(0.05, 4.5, length.out = 200),
  Ginf = seq(0.02, 0.30, length.out = 200)
) |>
  mutate(S_star = G2(g, Ginf) / Oij(g, Ginf))

p5 <- ggplot(grid5, aes(g, Ginf, fill = S_star)) +
  geom_raster() +
  geom_contour(aes(z = S_star), breaks = c(1, 2, 5, 10), color = "black", linewidth = 0.3) +
  geom_point(data = tibble(g = c(g0, g_PV), Ginf = c(Ginf0, Ginf0)),
             aes(g, Ginf), inherit.aes = FALSE, shape = 4, size = 3, stroke = 1.2) +
  scale_fill_viridis_c(trans = "log10") +
  labs(
    x        = "g_ij  (spike-dependent conductance, nS)",
    y        = "G_inf_ki(j)  (dendritic-segment admittance, nS)",
    fill     = "S* (log scale)",
    title    = "OLD rule's sign-flip threshold S* = G2(g,Ginf)/O(g,Ginf), over realistic (g, Ginf)",
    subtitle = "X marks = (g0=0.4, Ginf0=0.15) and (g_PV=4.0, Ginf0=0.15). Contours at S*=1,2,5,10."
  )
print(p5)

## =============================================================================
## OVERALL ASSESSMENT: IS THE (CORRECTED) NEW RULE AN IMPROVEMENT?
## =============================================================================
## More clearly yes than the previous (buggy) pass suggested. With eq. 36
## corrected:
##   + The forward-dynamics improvement from eq. 26 (saturating rather than
##     linear, so v_ij can no longer overshoot v_eq_ij) still holds.
##   + The derived rule is now algebraically CLEANER than the old rule -- it
##     is literally the old rule's first term, G2(g)*H(S), with no offset.
##     This is a simpler, more elegant result and a good sign that eq. 36 (as
##     corrected) is compatible with the g_pre/v_pre decomposition (eqs.
##     38-42): the exp(S*Gij) factors cancel exactly either way.
##   + The g -> 0 divergence introduced by the (now-fixed) bug is gone. Both
##     rules are bounded as conductance approaches zero.
##   + The new rule now VANISHES for strongly, persistently driven synapses
##     (S -> Infinity), rather than approaching a nonzero LTD floor like the
##     old rule does. Arguably more sensible: a synapse that is already
##     reliably transmitting shouldn't need much correction.
##   - It still loses the presynaptic-drive-dependent LTP/LTD sign flip that
##     the old rule had via O(g). Whether that switch was a desirable,
##     biologically-motivated feature of the old (linear) model or an
##     artifact of it is still an open question -- but note that the
##     corrected new rule's fixed sign now points the OPPOSITE way at low S
##     from what the (buggy) new rule previously reported: it depresses
##     (rather than potentiates) weakly-driven synapses when Hcab>0. This is
##     worth deciding deliberately rather than accepting by default.
##   - It still diverges as S -> 0 (shared with the old rule; not new).
##   - Hcab_ij's shift from I_syn,i to I_total,i (via eq. 42) is unchanged by
##     this fix and is still an open question worth resolving before any C++
##     implementation, since I_total,i includes the large transient
##     I_spike,i term.
##
## PARAMETER-SENSITIVITY FINDINGS (widened eta, and g/Ginf sweep, this pass)
## ------------------------------------------------------------------------------
## - eta and Hcab are purely multiplicative in both rules, so widening their
##   range (Plots 1-4) only ever mirrors existing curves through zero -- no
##   new curve shapes appear. This is a useful confirmation that the earlier,
##   eta>0-only plots weren't hiding anything qualitatively new, just half of
##   a symmetric picture.
## - g and Ginf are NOT purely multiplicative (they enter G2(g,Ginf) and, for
##   the old rule, O(g,Ginf) nonlinearly), and DO reveal something new: Plot 5
##   shows the old rule's threshold S* = G2(g,Ginf)/O(g,Ginf) is far from
##   constant across the network's actual cell-type range. At the biggest,
##   most proximal synapses (g=4.0 nS PV->ss, Ginf=0.15), S* ~ 1.04 (barely
##   above 1); at small, distal synapses (g=0.05 nS, Ginf=0.28), S* ~ 6.6; and
##   S* -> Infinity as g -> 0 for any Ginf. So the old rule's LTP/LTD boundary
##   isn't just "a threshold" -- it's a threshold that moves by an order of
##   magnitude or more depending on which synapse (cell-type pair, dendritic
##   location) you're looking at. Whether that's a desirable form of
##   heterogeneity or an unintended complication is unclear, but it's a
##   further point of complexity the new rule simply doesn't have, since its
##   sign never depends on g or Ginf at all.
##
## RECOMMENDATION: the rule is now simple and internally consistent. Before
## implementing it in C++, decide deliberately (1) whether the loss of the
## LTP/LTD sign switch (and its new, opposite-signed behavior at low S) is
## the intended behavior, (2) whether Hcab_ij should be built from I_syn,i or
## I_total,i, and (3) if the old rule's S* threshold is to be kept in some
## form, whether its cell-type/location-dependent variability (Plot 5) is
## intended or should be normalized out. The 1/S divergence at low
## presynaptic drive (shared with the old rule) will still need a floor on S
## for any online use.
## =============================================================================
## 
## Points favoring the new rule
## 
## It's the consistent consequence of a more biophysically grounded forward model. 
## Eq. 26's saturating form (v_ij bounded by v_eq) is basic electrochemistry — a 
## driving force can't push voltage past its own equilibrium point. The old 
## rule's linear v_ij could overshoot v_eq, which is not physically sensible. 
## Since the new Hebbian rule falls directly out of that better forward model, 
## it inherits that improvement.
## 
## It gives a single, coherent causal story: strengthen synapses whose marginal 
## contribution is currently reducing the cell's power, weaken those increasing 
## it — with no other confound. The old rule's extra S*-dependent flip means the 
## same "channel is hurting energy minimization" signal (eta>0) can still 
## produce LTP if S happens to be below threshold, which muddies that story.
## 
## A real point in the old rule's favor
##
## Bidirectional, threshold-dependent plasticity (low drive → LTD, high drive → LTP, 
## or vice versa) is a genuine, well-documented phenomenon — classic frequency-dependent 
## LTP/LTD experiments, and BCM theory's sliding threshold. The old rule's S*-crossing 
## structurally resembles this in a way the new rule (which never changes sign with S) 
## doesn't. That said, real metaplasticity thresholds (BCM) slide with postsynaptic 
## activity history, not with synapse conductance or dendritic admittance — so even if 
## the old rule looks superficially BCM-like, the mechanism setting its threshold doesn't 
## match the biology it resembles.
## 
## A flaw shared by both, and worth weighing more heavily than either point above
## 
## Both rules retain $H(S)\sim -1/S$, which means both get their effect size the wrong way round 
## relative to real STDP: magnitude grows the longer ago the presynaptic spike occurred (Plot 3), 
## rather than decaying with increasing pre/post separation as in every real STDP curve I know of. 
## Both also diverge at S→0 — a synapse with essentially no recent presynaptic activity gets the 
## largest possible update, which is backwards from any biological plasticity rule requiring some 
## minimal coincident activity to change at all.
## 
## An open question I can't resolve from what we've built
##
## Which sign of eta corresponds to "ordinary" synaptic transmission is genuinely undetermined 
## here — eta is treated as a free, exogenous axis in this whole analysis (nothing in the paper 
## ties its sign to whether a synapse is actively, successfully transmitting). So I'd be overclaiming 
## if I tried to say definitively "the new rule usually produces LTD/LTP for a normal active excitatory 
## synapse" — that depends on a mapping from network dynamics to eta's sign that hasn't actually been 
## established or simulated.
## 
## Net: the new rule is cleaner and better-grounded in the (now more realistic) forward dynamics, 
## and avoids an arbitrary, morphology-dependent threshold. But neither rule captures the hallmark 
## timing-decay of real STDP, and both share the same low-S divergence — so I'd call the new rule 
## "less wrong," not "biologically realistic" outright.