## =============================================================================
## arbor_spiking_check.py   (2026-11-20)
## Spiking-cell comparison of ways to carry dendritic current to the soma.
## Companion to arbor_sweep_notes.tex (paper_DACxmini), section "Numerical check".
##
## Cell: soma --[k]-- junction --[c1]-- tip1 (synapse, E = 0)
##                             \-[c2]-- tip2 (sealed)
## The soma is a growth-transform (GT) neuron with the all-Ohmic power function
## of the paper: leak, a fixed tonic drive, a threshold-gated spike channel with
## reversal -V_bound, reset by the full attractor step, 1 ms hold.
##
## Methods compared
##   REF      Soma is a GT node. The whole dendrite is a fine passive cable WITH
##            membrane capacitance (time constant tau_c, set equal to the soma's
##            effective membrane time constant), solved implicitly.
##   REF fast The same with tau_c at 1 percent: a cable that settles almost at once.
##   E0       End-potential scheme (two-port coefficients); dendritic points are
##            solved algebraically each step (no inertia). Should equal REF fast.
##   E1       End-potential scheme; dendritic points are GT nodes with the same
##            Lambda and rate as the soma.
##   E1+d     E1 with the far-end potential delayed by L * tau_c / 2 per segment.
##   Ef+d     As E1+d, but the dendritic points relax ten times faster.
##   E2       E1 with each dendritic point's rate set so that its effective
##            capacitance equals that of the membrane next to it.
##   W1       Notebook sweep as written: each node's potential is set by its own
##            local currents plus the attenuated current from its children, and
##            that same current is what it hands to its parent. Potentials are
##            held at the tip, the branch point and the soma. Each dendritic
##            node carries a leak equal to half the membrane conductance of each
##            attached segment. The soma load is the sum of its open
##            conductances (I_soma/V_soma is not usable).
##   W2       W1 with the current split at the branch point.
##
## Output per run: soma firing rate, the time-averaged depolarizing current the
## dendrite delivers to the soma, the mean tip potential, and the largest
## |2I|/Lambda on steps where the attractor is used (must stay below 1).
##
## Findings (2026-11-20):
##  - W1/W2 hand the soma a current that averages to zero, and the soma never
##    feels the dendrite as a load. With a steady tip synapse the firing rate
##    does not depend on the synapse at all.
##  - E0 reproduces REF fast. E1 is within about 10 percent of REF for a steady
##    synapse and 10 to 20 percent high for brief pulses (more near threshold).
##  - The delay on top of GT inertia at the dendritic points overcorrects.
##    Fast dendritic points plus the delay do about as well as E1.
##
## Run: python3 arbor_spiking_check.py     (numpy only, about 7 minutes)
## =============================================================================
import math, sys
import numpy as np

P = dict(Vb=90.0, thr=-50.0, Vr=-70.0, gl=10.0, gs=100.0, lam_mult=1.02, tau_fast=5.0, hold=1.0)
SEG = {"k": (0.5, 10.0), "c1": (0.8, 6.0), "c2": (0.6, 5.0)}   # (electrotonic length, G_inf nS)
ESYN = 0.0
GD = 6.0            # fixed tonic excitatory conductance at the soma (nS, E = 0)
TAU_SYN = 2.0
DT = 0.01
LAM = 2 * P["gs"] * (P["thr"] + P["Vb"]) * P["lam_mult"]

def ceff(V):        # effective capacitance of a GT node (pF) for small currents
    return P["Vb"] * LAM * P["tau_fast"] / (2 * (P["Vb"] ** 2 - V ** 2))
TAU_M_SOMA = ceff(-60.0) / P["gl"]      # about 41 ms

def gt_star(V, I2):
    Vb = P["Vb"]
    den = Vb * LAM - V * I2
    return Vb * (V * LAM - Vb * I2) / den, den

class Soma:
    def __init__(s):
        s.V = P["Vr"]; s.hold = 0; s.nhold = int(round(P["hold"] / DT)); s.spikes = []; s.bad = 0; s.maxr = 0.0
    def local(s):
        """Sum of the soma's own currents (pA) and whether the spike gate is on."""
        on = s.V >= P["thr"]
        I = P["gl"] * (s.V - P["Vr"]) + GD * (s.V - 0.0)
        if on: I += P["gs"] * (s.V + P["Vb"])
        return I, on
    def step(s, Itot, on, k):
        I2 = 2 * Itot
        Vs, den = gt_star(s.V, I2)
        if on or s.hold == 0:           # the attractor is not used during the hold
            s.maxr = max(s.maxr, abs(I2) / LAM)
            if den <= 0: s.bad += 1
        if on:
            s.V = Vs; s.hold = s.nhold; s.spikes.append(k)
        elif s.hold > 0:
            s.hold -= 1
        else:
            s.V = min(s.V + (Vs - s.V) * DT / P["tau_fast"], P["thr"])

def gt_node(V, Itot, rate=1.0):
    Vs, _ = gt_star(V, 2 * Itot)
    return V + (Vs - V) * rate * DT / P["tau_fast"]

def gate_fn(mode, period):
    """tonic: S = 1 always. pulsed: S set to 1 every `period` ms, decays with TAU_SYN."""
    if mode == "tonic":
        return lambda k, S: 1.0
    n = int(round(period / DT)); dec = math.exp(-DT / TAU_SYN)
    return lambda k, S: 1.0 if k % n == 0 else S * dec

def Yself(name):  L, G = SEG[name]; return G / math.tanh(L)
def Ytr(name):    L, G = SEG[name]; return G / math.sinh(L)
def Gin(name, load):
    L, G = SEG[name]; return G * (load + G * math.tanh(L)) / (G + load * math.tanh(L))
def Aup(name, load):
    L, G = SEG[name]; return math.cosh(L) + (G / load) * math.sinh(L)

def summarize(soma, T, burn, Idel, Vtip):
    sp = [k * DT for k in soma.spikes if k * DT >= burn]
    rate = 1000 * (len(sp) - 1) / (sp[-1] - sp[0]) if len(sp) > 1 else 0.0
    return dict(rate=rate, Idel=Idel, Vtip=Vtip, bad=soma.bad, maxr=soma.maxr)

# ---------------------------------------------------------------- REF
def run_ref(g, mode, period, tau_c, T, burn, n=40):
    Vr = P["Vr"]; idx = {}; N = 0
    for name in SEG:
        idx[name] = list(range(N, N + n)); N += n
    # point n of k is the junction; c1 and c2 start at the junction
    junc = idx["k"][-1]; tip = idx["c1"][-1]; k1 = idx["k"][0]
    Gm = np.zeros((N, N)); gmem = np.zeros(N)
    ga_soma = None; gm_soma_half = None
    for name, (L, Ginf) in SEG.items():
        ga = Ginf * n / L; gm = Ginf * L / n
        chain = ([None] if name == "k" else [junc]) + idx[name]      # None = soma
        for i in range(n):
            p, q = chain[i], chain[i + 1]
            Gm[q, q] += ga; gmem[q] += gm / 2
            if p is None:
                ga_soma = ga; gm_soma_half = gm / 2
            else:
                Gm[p, p] += ga; Gm[p, q] -= ga; Gm[q, p] -= ga; gmem[p] += gm / 2
    Gm += np.diag(gmem)
    C = tau_c * gmem
    # Backward Euler: (C/dt + Gm + g S e e^T) u_new = (C/dt) u + inputs. The matrix without the
    # synapse is fixed, so work in its eigenbasis (one multiply per step) and add the synapse,
    # a rank-one change at the tip, with the Sherman-Morrison formula.
    Ainv = np.linalg.inv(np.diag(C / DT) + Gm)
    Dh = np.sqrt(C / DT)
    lam, Q = np.linalg.eigh(Dh[:, None] * Ainv * Dh[None, :])
    p1 = Q.T @ (Dh * Ainv[:, k1]); p2 = Q.T @ (Dh * Ainv[:, tip]); att = Ainv[tip, tip]
    r_k1 = Q[k1, :] / Dh[k1]; r_tip = Q[tip, :] / Dh[tip]
    z = np.zeros(N); u1 = ut = 0.0
    soma = Soma(); gate = gate_fn(mode, period); S = 0.0
    sI = sV = 0.0; cnt = 0
    for k in range(int(T / DT)):
        S = gate(k, S); gs_ = g * S
        vs = soma.V - Vr
        Iloc, on = soma.local()
        Iax = ga_soma * (vs - u1) + gm_soma_half * vs
        z0 = lam * z + p1 * (ga_soma * vs) + p2 * (gs_ * (ESYN - Vr))
        z = z0 - p2 * (gs_ * float(r_tip @ z0) / (1 + gs_ * att))
        u1 = float(r_k1 @ z); ut = float(r_tip @ z)
        soma.step(Iloc + Iax, on, k)
        if k * DT >= burn:
            sI += -Iax; sV += ut + Vr; cnt += 1
    return summarize(soma, T, burn, sI / cnt, sV / cnt)

# ---------------------------------------------------------------- E0, E1, E1d
def run_end(g, mode, period, T, burn, kind="E1", delay_tau=None, rate_j=1.0, rate_t=1.0):
    Vr = P["Vr"]
    Yk, Tk, Y1, T1 = Yself("k"), Ytr("k"), Yself("c1"), Ytr("c1")
    G2 = Gin("c2", 0.0)                       # sealed side branch folded into the junction
    soma = Soma(); vj = vt = 0.0; gate = gate_fn(mode, period); S = 0.0
    if delay_tau:
        dk = int(round(SEG["k"][0] * delay_tau / 2 / DT)); d1 = int(round(SEG["c1"][0] * delay_tau / 2 / DT))
        nb = max(dk, d1) + 1
        hs = np.zeros(nb); hj = np.zeros(nb); ht = np.zeros(nb)
    sI = sV = 0.0; cnt = 0
    for k in range(int(T / DT)):
        S = gate(k, S); gs_ = g * S
        vs = soma.V - Vr
        Iloc, on = soma.local()
        if kind == "E0":
            # dendritic points settle at once: solve I_j = 0 and I_t = 0 for vj, vt
            a11 = Yk + Y1 + G2; a22 = Y1 + gs_
            det = a11 * a22 - T1 * T1
            r1 = Tk * vs; r2 = gs_ * (ESYN - Vr)
            vj = (r1 * a22 + T1 * r2) / det; vt = (a11 * r2 + T1 * r1) / det
            Jsj = Yk * vs - Tk * vj
        else:
            if delay_tau:
                hs[k % nb] = vs; hj[k % nb] = vj; ht[k % nb] = vt
                vs_k = hs[(k - dk) % nb]; vj_k = hj[(k - dk) % nb]
                vj_1 = hj[(k - d1) % nb]; vt_1 = ht[(k - d1) % nb]
            else:
                vs_k, vj_k, vj_1, vt_1 = vs, vj, vj, vt
            Jsj = Yk * vs - Tk * vj_k
            Ij = (Yk * vj - Tk * vs_k) + (Y1 * vj - T1 * vt_1) + G2 * vj
            It = gs_ * (vt + Vr - ESYN) + (Y1 * vt - T1 * vj_1)
            vj = gt_node(vj + Vr, Ij, rate_j) - Vr
            vt = gt_node(vt + Vr, It, rate_t) - Vr
        soma.step(Iloc + Jsj, on, k)
        if k * DT >= burn:
            sI += -Jsj; sV += vt + Vr; cnt += 1
    return summarize(soma, T, burn, sI / cnt, sV / cnt)

# ---------------------------------------------------------------- W1, W2
def run_sweep(g, mode, period, T, burn, split=False, leak_scale=1.0):
    Vr = P["Vr"]
    # lumped membrane leak at the dendritic nodes: half of each attached segment
    half = {nm: SEG[nm][0] * SEG[nm][1] / 2 for nm in SEG}
    g_lt = leak_scale * half["c1"]
    g_lj = leak_scale * (half["c1"] + half["c2"] + half["k"])
    soma = Soma(); Vj = Vt = Vr; gate = gate_fn(mode, period); S = 0.0
    sI = sV = 0.0; cnt = 0
    for k in range(int(T / DT)):
        S = gate(k, S); gs_ = g * S
        Iloc, on = soma.local()
        load = P["gl"] + GD + (P["gs"] if on else 0.0)
        Gk = Gin("k", load); Gup1 = Gk + Gin("c2", 0.0)
        It = gs_ * (Vt - ESYN) + g_lt * (Vt - Vr)
        Ic1 = It / Aup("c1", Gup1)
        Ij = g_lj * (Vj - Vr) + Ic1
        Ipass = (g_lj * (Vj - Vr) + Ic1 * (Gk / Gup1 if split else 1.0)) / Aup("k", load)
        Vt = gt_node(Vt, It); Vj = gt_node(Vj, Ij)
        soma.step(Iloc + Ipass, on, k)
        if k * DT >= burn:
            sI += -Ipass; sV += Vt; cnt += 1
    return summarize(soma, T, burn, sI / cnt, sV / cnt)

# ---------------------------------------------------------------- main
if __name__ == "__main__":
    T, burn = 4000.0, 1000.0
    tc = TAU_M_SOMA
    half = {nm: SEG[nm][0] * SEG[nm][1] / 2 for nm in SEG}      # half the membrane conductance of each segment
    rt = ceff(-60.0) / (tc * half["c1"])                         # tip: half of c1
    rj = ceff(-60.0) / (tc * (half["c1"] + 2 * half["c2"] + half["k"]))   # junction: half of c1 and k, all of c2
    print(f"Lambda = {LAM:.0f} pA, soma effective membrane time constant = tau_c = {tc:.1f} ms")
    methods = [
        ("REF",      lambda g, m, p: run_ref(g, m, p, tc, T, burn)),
        ("REF fast", lambda g, m, p: run_ref(g, m, p, 0.01 * tc, T, burn)),
        ("E0",       lambda g, m, p: run_end(g, m, p, T, burn, "E0")),
        ("E1",       lambda g, m, p: run_end(g, m, p, T, burn)),
        ("E1+d",     lambda g, m, p: run_end(g, m, p, T, burn, delay_tau=tc)),
        ("Ef+d",     lambda g, m, p: run_end(g, m, p, T, burn, delay_tau=tc, rate_j=10.0, rate_t=10.0)),
        ("E2",       lambda g, m, p: run_end(g, m, p, T, burn, rate_j=rj, rate_t=rt)),
        ("W1",       lambda g, m, p: run_sweep(g, m, p, T, burn, False)),
        ("W2",       lambda g, m, p: run_sweep(g, m, p, T, burn, True)),
    ]
    blocks = [(6.0, [("tonic", None, g) for g in (0.0, 2.0, 5.0, 10.0, 20.0)] + [("pulsed", 20.0, 50.0)]),
              (8.0, [("tonic", None, 0.0), ("pulsed", 20.0, 20.0), ("pulsed", 20.0, 50.0)])]
    for gd, cases in blocks:
        GD = gd
        for mode, period, g in cases:
            tag = f"soma drive {gd} nS; {mode} tip synapse, g = {g} nS" + (f", every {period} ms" if period else "")
            print(f"\n--- {tag}")
            print("  method     rate (Hz)   mean depol. current into soma (pA)   mean V_tip (mV)   max|2I|/Lambda  bad")
            for name, f in methods:
                r = f(g, mode, period)
                print(f"  {name:9s}  {r['rate']:8.2f}   {r['Idel']:12.2f}                      {r['Vtip']:8.2f}        {r['maxr']:6.3f}    {r['bad']}")
                sys.stdout.flush()
