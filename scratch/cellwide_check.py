# Numerical check of a single cell-wide power function for GT nodes joined by passive stretches.
import math, numpy as np
rng=np.random.default_rng(1)
Vb=90.0; Vr=-70.0
def stretch(L,G): return G/math.tanh(L), G/math.sinh(L)       # Yself, Ytr (uniform cylinder)
def build(stretches, n, gnode):
    """stretches: list of (a,b,L,Ginf). gnode: list per node of [(g,E),...]. Returns A (n x n), c (n): grad = 2(A V - c)."""
    A=np.zeros((n,n)); c=np.zeros(n)
    for a,lst in enumerate(gnode):
        for g,E in lst: A[a,a]+=g; c[a]+=g*E
    for a,b,L,G in stretches:
        Ys,Yt=stretch(L,G)
        A[a,a]+=Ys; A[b,b]+=Ys; A[a,b]-=Yt; A[b,a]-=Yt
        c[a]+=(Ys-Yt)*Vr; c[b]+=(Ys-Yt)*Vr
    return A,c
def H(V,stretches,gnode):
    h=sum(g*(V[a]-E)**2 for a,lst in enumerate(gnode) for g,E in lst)
    for a,b,L,G in stretches:
        Ys,Yt=stretch(L,G); va=V[a]-Vr; vb=V[b]-Vr
        h+=Ys*va*va-2*Yt*va*vb+Ys*vb*vb
    return h
def Hnode(a,V,stretches,gnode):          # per-node function with neighbours held fixed (eq. 6 of the summary)
    h=sum(g*(V[a]-E)**2 for g,E in gnode[a])
    for x,y,L,G in stretches:
        if a in (x,y):
            b=y if a==x else x; Ys,Yt=stretch(L,G)
            h+=Ys*(V[a]-(Vr+(Yt/Ys)*(V[b]-Vr)))**2
    return h
def gt_step(V,Gd,Lam,r):
    Vs=Vb*(V*Lam-Vb*Gd)/(Vb*Lam-V*Gd)
    return V+r*(Vs-V)
# ---- test tree of the spiking check: soma(0) - k - branch(1) - c1 - tip(2, synapse) ; branch - c2 - tip(3)
ST=[(0,1,0.5,10.0),(1,2,0.8,6.0),(1,3,0.6,5.0)]
def gn(gsyn,spike=False):
    s=[(10.0,Vr),(6.0,0.0)]+([(100.0,-Vb)] if spike else [])
    return [s,[],[(gsyn,0.0)],[]]
A,c=build(ST,4,gn(10.0)); V=np.array([-62.0,-60.0,-55.0,-61.0])
g_fd=np.array([(H(V+e,ST,gn(10.0))-H(V-e,ST,gn(10.0)))/2e-4 for e in 1e-4*np.eye(4)])
print("1. gradient: max |finite difference - 2(AV-c)| =", float(np.max(np.abs(g_fd-2*(A@V-c)))))
I0=10*(V[0]-Vr)+6*V[0]+ (stretch(0.5,10)[0]*(V[0]-Vr)-stretch(0.5,10)[1]*(V[1]-Vr))
print("   node rule at soma: 2*I_a =",2*I0," gradient =",2*(A@V-c)[0])
print("   eigenvalues of A:",np.round(np.linalg.eigvalsh(A),3))
# ---- 2. descent run between spikes with different rates and bounds per node
Rsum=np.sum(np.abs(A),axis=1)
r=np.array([0.002,0.02,0.2,1.0])               # tau_V*dt per node, deliberately very different
V=np.array([-70.0,-70.0,-70.0,-70.0]); Hs=[]; up_node=np.zeros(4,int)
Vmin=np.linalg.solve(A,c)
for k in range(4000):
    Gd=2*(A@V-c); Lam=np.maximum(1.02*np.abs(Gd),2*Vb*r*Rsum)+1e-9
    Vn=gt_step(V,Gd,Lam,r)
    for a in range(4):
        if Hnode(a,Vn,ST,gn(10.0))>Hnode(a,V,ST,gn(10.0))+1e-9: up_node[a]+=1
    Hs.append(H(V,ST,gn(10.0))); V=Vn
Hs=np.array(Hs)
print("2. steps where cell-wide H rose:",int(np.sum(np.diff(Hs)>1e-9)),"of",len(Hs)-1,"| H start/end:",round(Hs[0],2),round(Hs[-1],2),"| H at A V = c:",round(H(Vmin,ST,gn(10.0)),2))
print("   final V:",np.round(V,3)," steady state:",np.round(Vmin,3))
print("   steps where a node's OWN per-node function rose (same run):",up_node.tolist())
# ---- 3. random trials
def trial(mode):
    n=int(rng.integers(3,9)); st=[(int(rng.integers(0,b)),b,float(rng.uniform(0.02,1.5)),float(rng.uniform(1,20))) for b in range(1,n)]
    g=[[(float(rng.uniform(0,30)),float(rng.uniform(-Vb,Vb)))]+[(float(rng.uniform(0,10)),Vr)] for _ in range(n)]
    A,c=build(st,n,g); V=rng.uniform(-0.98*Vb,0.98*Vb,n); r=rng.uniform(0.01,1.0,n)
    Gd=2*(A@V-c); Rs=np.sum(np.abs(A),axis=1)
    Lam=1.02*np.abs(Gd)+1e-9 if mode=="grad" else np.maximum(1.02*np.abs(Gd),2*Vb*r*Rs)+1e-9
    Vn=gt_step(V,Gd,Lam,r); return H(Vn,st,g)-H(V,st,g), bool(np.all(np.abs(Vn)<Vb))
for mode in ("both","grad"):
    res=[trial(mode) for _ in range(20000)]
    print("3. random cells, bound =",{"both":"gradient bound and coupling bound","grad":"gradient bound only"}[mode],": H rose in",sum(d>1e-9 for d,_ in res),"of",len(res),"| stayed inside bounds:",all(ok for _,ok in res))
# ---- 4. spike reset step in the test tree (soma at threshold, spike gate on, soma takes a full step)
g=gn(10.0,spike=True); A,c=build(ST,4,g); LAM=2*100.0*(-50.0+Vb)*1.02
V=np.linalg.solve(*build(ST,4,gn(10.0))); V=V+( -50.0 - V[0])*np.array([1,.8,.6,.7])   # a state with the soma at threshold
Gd=2*(A@V-c); r=np.array([1.0,0.002,0.002,0.002]); Rs=np.sum(np.abs(A),axis=1)
Vn=gt_step(V,Gd,np.full(4,LAM),r)
print("4. reset step: soma",round(V[0],2),"->",round(Vn[0],2),"| cell-wide H change:",round(H(Vn,ST,g)-H(V,ST,g),1),"| Lambda =",LAM,"| sufficient bound at soma 2*Vb*R =",round(2*Vb*Rs[0],0),"| |grad|/Lambda at soma:",round(abs(Gd[0])/LAM,3))
# ---- 5. which condition separates the cases
def trial2():
    n=int(rng.integers(3,9)); st=[(int(rng.integers(0,b)),b,float(rng.uniform(0.02,1.5)),float(rng.uniform(1,20))) for b in range(1,n)]
    g=[[(float(rng.uniform(0,30)),float(rng.uniform(-Vb,Vb)))]+[(float(rng.uniform(0,10)),Vr)] for _ in range(n)]
    A,c=build(st,n,g); V=rng.uniform(-0.98*Vb,0.98*Vb,n); r=rng.uniform(0.01,1.0,n)
    Gd=2*(A@V-c); Rs=np.sum(np.abs(A),axis=1); Lam=1.02*np.abs(Gd)+1e-9
    inv_m=(Vb*Lam-V*Gd)/(r*(Vb**2-V**2))                      # 1/(r_a eta_a)
    local=bool(np.all(inv_m>Rs)); exact=bool(np.min(np.linalg.eigvalsh(np.diag(inv_m)-A))>0)
    box=2*(Vb*Rs+np.abs(c))                                    # largest |gradient| over the whole box
    d1=H(gt_step(V,Gd,Lam,r),st,g)-H(V,st,g); d2=H(gt_step(V,Gd,box,r),st,g)-H(V,st,g)
    return d1>1e-9,local,exact,d2>1e-9
res=[trial2() for _ in range(20000)]
rose=sum(a for a,_,_,_ in res)
print("5. gradient bound only:",rose,"rises;",sum(a and l for a,l,_,_ in res),"of them with the local check satisfied;",sum(a and e for a,_,e,_ in res),"with the exact condition satisfied")
print("   cases passing the local check:",sum(l for _,l,_,_ in res)," passing the exact condition:",sum(e for _,_,e,_ in res))
print("   Lambda = largest gradient over the box: H rose in",sum(d for _,_,_,d in res),"of",len(res))
# ---- 6. the reset step and a between-spike step of the test tree against the local check
for label,spike,r in (("reset step",True,np.array([1.0,0.002,0.002,0.002])),("between spikes",False,np.full(4,0.002))):
    g=gn(10.0,spike=spike); A,c=build(ST,4,g); Rs=np.sum(np.abs(A),axis=1)
    V=np.array([-50.0,-52.0,-40.0,-54.0]) if spike else np.array([-60.0,-59.0,-45.0,-60.0])
    Gd=2*(A@V-c); inv_m=(Vb*LAM-V*Gd)/(r*(Vb**2-V**2))
    print("6.",label,": 1/(r*eta) per node (nS) =",np.round(inv_m,0),"| row sums R (nS) =",np.round(Rs,1))
