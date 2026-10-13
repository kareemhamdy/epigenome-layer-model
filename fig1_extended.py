"""
Figure 1 (extended) -- scalar positive-feedback aging model. ILLUSTRATIVE: parameters
chosen to show a fold is POSSIBLE, not fitted to data.

    dD/dt = u*(1 + h*D^n/(K^n + D^n)) - gamma*A*D ,   A = a1*a2

Arms (each intervention adds delta = 0.12 to capacity, baseline a1 = a2 = sqrt(A0)):
  cross-layer : agent X raises a1, agent Y raises a2 (independent primitives).
  same-layer  : X and Y both raise a1 (one capacity term).
  shared-pool : X and Y both add supply s to ONE saturable pool N; both primitives
                read it, a_i = N/(N+kappa). Each dose is calibrated so that, at baseline,
                it raises a_i by delta.
Two statistics, both computed by relaxing from the AGED (high-D) steady state:
  Bliss excess = E_pair - (E1 + E1 - E1*E1)
  Advantage    = E_pair - max(E_single_at_double_dose, E_single_at_single_dose)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

u, h, K, n, gamma = 0.02, 8.0, 0.3, 6, 1.0
delta, kappa = 0.12, 1.0
HIGH_START, T_SETTLE = 2.0, 8000.0

def rhs(t, y, A):
    D = max(y[0], 0.0)
    return [u*(1 + h*D**n/(K**n + D**n)) - gamma*A*D]

def settle(D0, A, T=T_SETTLE):
    return solve_ivp(rhs, [0, T], [D0], args=(A,), method="LSODA",
                     rtol=1e-9, atol=1e-12).y[0, -1]

def steady_states(A):
    c = np.zeros(n + 2)
    c[0], c[1], c[n], c[n+1] = gamma*A, -u*(1+h), gamma*A*K**n, -u*K**n
    r = np.roots(c); r = np.real(r[np.abs(np.imag(r)) < 1e-9])
    return np.sort(r[r > 0])

# ---- capacities for each arm: (single dose, pair, double-dose single) -> A values
def arm_capacities(a, arm):
    if arm == "cross":
        return (a+delta)*a, (a+delta)**2, (a+2*delta)*a
    if arm == "same":
        return (a+delta)*a, (a+2*delta)*a, (a+2*delta)*a
    if arm == "shared":
        N0 = kappa*a/(1-a)
        s = kappa*((a+delta)/(1-a-delta) - a/(1-a))
        g = lambda N: N/(N+kappa)
        return g(N0+s)**2, g(N0+2*s)**2, g(N0+2*s)**2

def stats(A0, arm):
    a = np.sqrt(A0)
    Db = settle(HIGH_START, A0)
    if Db <= 1e-9:
        return 0.0, 0.0
    A1, Ap, A2 = arm_capacities(a, arm)
    E = lambda A: 1 - settle(Db, A)/Db
    E1, Ep, E2 = E(A1), E(Ap), E(A2)
    return Ep - (2*E1 - E1*E1), Ep - max(E2, E1)

A0s = np.linspace(0.04, 0.50, 300)
res = {arm: np.array([stats(A0, arm) for A0 in A0s]) for arm in ("cross", "same", "shared")}

# Panel A (fold) and Panel C (hysteresis)
A_vals = np.linspace(0.02, 0.60, 1000)
low, mid, high = [], [], []
for A in A_vals:
    r = steady_states(A)
    if len(r) >= 3:
        low.append((A, r[0])); mid.append((A, r[1])); high.append((A, r[-1]))
    elif len(r) == 1:
        (low if r[0] < K else high).append((A, r[0]))
A_sw = np.linspace(0.05, 0.55, 160)
Dn = settle(HIGH_START, A_sw[0]); fwd = []
for A in A_sw: Dn = settle(Dn, A); fwd.append(Dn)
Dn = fwd[-1]; bwd = []
for A in A_sw[::-1]: Dn = settle(Dn, A); bwd.append(Dn)
bwd = bwd[::-1]

fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))
ax = ax.ravel()
for data, c, s, lab in [(low, "tab:blue", "-", r"$D_{low}$ (stable)"),
                        (high, "tab:red", "-", r"$D_{high}$ (stable)"),
                        (mid, "k", "--", r"$D_{saddle}$ (unstable)")]:
    if data:
        arr = np.array(data); ax[0].plot(arr[:, 0], arr[:, 1], color=c, ls=s, lw=2, label=lab)
ax[0].set(xlabel=r"Effective capacity $\gamma A$", ylabel=r"Damage burden $D^*$", title="A. Bifurcation (fold)")
ax[0].legend(); ax[0].grid(alpha=.3)

sty = {"cross": ("purple", "-", "Cross-layer"), "same": ("tab:orange", "--", "Same-layer"),
       "shared": ("tab:green", ":", "Shared pool")}
for arm, (c, ls, lab) in sty.items():
    ax[1].plot(A0s, res[arm][:, 0], color=c, ls=ls, lw=2.2, label=lab)
ax[1].axhline(0, color="gray", ls=":", lw=1)
ax[1].set(xlabel=r"Baseline capacity $\gamma A_0$", ylabel="Excess over Bliss",
          title="B. Excess over Bliss (from aged state)")
ax[1].legend(); ax[1].grid(alpha=.3)

ax[2].plot(A_sw, fwd, "g-", lw=2, label=r"Forward (rising $A$)")
ax[2].plot(A_sw, bwd, color="orange", ls="--", lw=2, label=r"Backward (falling $A$)")
ax[2].set(xlabel=r"Effective capacity $\gamma A$", ylabel=r"Damage burden $D^*$", title="C. Hysteresis loop")
ax[2].legend(); ax[2].grid(alpha=.3)

for arm, (c, ls, lab) in sty.items():
    ax[3].plot(A0s, res[arm][:, 1], color=c, ls=ls, lw=2.2, label=lab)
ax[3].axhline(0, color="gray", ls=":", lw=1)
ax[3].set(xlabel=r"Baseline capacity $\gamma A_0$", ylabel="Advantage over double-dose single agent",
          title="D. Cross-layer advantage (equal total dose)")
ax[3].legend(); ax[3].grid(alpha=.3)
plt.tight_layout()
plt.savefig("Figure1_extended.png", dpi=300, bbox_inches="tight")
plt.savefig("Figure1_extended.pdf", bbox_inches="tight")

band = [A for A in A_vals if len(steady_states(A)) >= 3]
print(f"Bistable band: [{min(band):.3f}, {max(band):.3f}]")
for arm in res:
    ex, adv = res[arm][:, 0], res[arm][:, 1]
    pos = A0s[ex > 0.05]; neg = A0s[ex < -0.01]
    ap = A0s[adv > 0.05]
    print(f"{arm:7s} Bliss>0.05: {(round(pos.min(),3), round(pos.max(),3)) if len(pos) else None} "
          f"peak {ex.max():+.3f} | Bliss<-0.01: {(round(neg.min(),3), round(neg.max(),3)) if len(neg) else None} "
          f"min {ex.min():+.3f} | Adv>0.05: {(round(ap.min(),3), round(ap.max(),3)) if len(ap) else None} "
          f"peak {adv.max():+.3f}; max|adv| {np.abs(adv).max():.3f}")
