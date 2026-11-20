## =============================================================================
## scratch/timed_gate_power_check.py   (2026-11-20)
## Does a TIMED synaptic gate lower average dissipated power?
##
## Companion to hebbian_stdp_toy.R. That script measures spike_cost, the
## numerator of the growth-transform attractor ratio. This one measures the
## dissipated power H itself, using the all-Ohmic form in the DACxmini paper
## (eq:Power): H = sum_k g_k (V - E_k)^2, with a threshold-gated spike channel
## whose reversal is -V_bound.
##
## Setup: one GT neuron under a tonic fixed excitatory conductance, so it fires
## periodically. A test synapse (excitatory, E = 0, or inhibitory, E = -80) has
## its gate set to 1 at a fixed delay after each postsynaptic spike and then
## decays with tau_syn. In the periodic state the gate therefore opens at a
## fixed lag before the NEXT postsynaptic spike.
##
## Simplifications: soma synapse (no attenuation), plain Ohmic synaptic current
## (no saturating v_syn), no calcium dynamics, two parameter sets.
##
## Averages run over a whole number of spike cycles (spike to spike), so
## window-edge effects cannot fake a difference.
##
## Output columns, per lag: change in average power vs no test synapse (fW),
## split into the direct cost of the open channel and the indirect remainder,
## plus the change in firing rate.
##
## Finding: an excitatory gate that opens within about 1 to 2 ms before the
## postsynaptic spike LOWERS average power (shallower reset, cheaper recovery);
## at other lags it raises power. An inhibitory gate mirrors this: it raises
## power when open at the postsynaptic spike and lowers it when it arrives a
## few ms earlier and delays the spike.
##
## Run: python3 scratch/timed_gate_power_check.py   (about a minute, no deps)
## =============================================================================
import math

def run(P, g, d, Es=0.0, T=8000.0, ncyc=20, dt=0.01):
    Vb, thr, Vr = P["Vb"], P["thr"], P["Vr"]
    gl, gs, gd = P["gl"], P["gs"], P["gd"]
    LAM = 2 * gs * (thr + Vb) * P["lam_mult"]
    decay = math.exp(-dt / P["tau_syn"])
    nhold = int(round(P["hold"] / dt))
    V, S, hold = Vr, 0.0, 0
    onset = -1
    E = Ed = 0.0
    spk = []      # (step, cumulative E, cumulative direct E) at each post spike
    pre = []
    for k in range(int(T / dt)):
        S *= decay
        if k == onset:
            S = 1.0; pre.append(k)
        spike = V >= thr
        if spike: spk.append((k, E, Ed))
        gp = g * S
        H = gd * V * V + gp * (V - Es) ** 2 + gl * (V - Vr) ** 2
        I = 2 * (gd * V + gp * (V - Es) + gl * (V - Vr))
        if spike:
            H += gs * (V + Vb) ** 2
            I += 2 * gs * (V + Vb)
        Vstar = Vb * (V * LAM - Vb * I) / (Vb * LAM - V * I)
        E += H * dt; Ed += gp * (V - Es) ** 2 * dt
        if spike:
            V = Vstar; hold = nhold
            if d is not None: onset = k + int(round(d / dt))
        elif hold > 0:
            hold -= 1
        else:
            V = min(V + (Vstar - V) * dt / P["tau_fast"], thr)
    a, b = spk[-1 - ncyc], spk[-1]
    Tm = (b[0] - a[0]) * dt
    # lag from each pre onset to the next post spike, within the measured stretch
    lags = []
    st = [s[0] for s in spk]
    for p in pre:
        if p >= a[0] and p < b[0]:
            nxt = next(s for s in st if s > p)
            lags.append((nxt - p) * dt)
    lag = sum(lags) / len(lags) if lags else float("nan")
    return (b[1] - a[1]) / Tm, 1000 * ncyc / Tm, (b[2] - a[2]) / Tm, lag, len(lags) / ncyc

sets = {
 "R-script-like": dict(Vb=80.5, thr=-55.0, Vr=-70.0, gl=10.0, gs=1000/25.5, gd=3.5,
                       lam_mult=1.05, tau_syn=2.0, tau_fast=5.0, hold=1.0),
 "plan-toy-like": dict(Vb=90.0, thr=-50.0, Vr=-70.0, gl=10.0, gs=100.0, gd=5.0,
                       lam_mult=1.02, tau_syn=2.0, tau_fast=5.0, hold=1.0),
}

for name, P in sets.items():
    H0, r0, _, _, _ = run(P, 0.0, None)
    isi0 = 1000 / r0
    print(f"\n=== {name}: baseline H_avg={H0:.2f} fW  rate={r0:.3f} Hz  ISI={isi0:.2f} ms")
    for label, Es, g in (("EXC E=0", 0.0, 0.1), ("INH E=-80", -80.0, 0.1), ("INH E=-80", -80.0, 1.0)):
        print(f"  {label}, g={g} nS:  lag pre->post(ms)  gate/cyc   dH_avg  = direct + indirect   d(rate Hz)")
        for f in (0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995):
            H, r, D, lag, frac = run(P, g, f * isi0, Es)
            print(f"      {lag:8.2f}   {frac:5.2f}   {H-H0:+9.3f}  {D:+8.3f}  {H-H0-D:+9.3f}   {r-r0:+7.3f}")
