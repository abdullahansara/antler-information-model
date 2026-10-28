# -*- coding: utf-8 -*-
"""Sample size for P3 (diagnosed injury-origin branch vs quality), individual males as the unit of analysis.
Design: each male is followed for up to 6 seasons (per-season survival 0.7, mean 2.94 seasons observed); a season diagnosed with probability pi;
a male counts as 'diagnosed' if diagnosed in any observed season. Quality is a stable individual trait, so seasons of one male are not independent
observations of the contrast; the unit is the male. True injury-origin males differ by d SD in quality; a fraction PPV of diagnosed males is truly
injury-origin, so the diagnosed-vs-undiagnosed difference is PPV*d. Quality is measured with reliability rel (error variance (1-rel)/rel).
Power is computed from the normal approximation to the two-sample test (alpha 0.05, two-sided) and checked by simulation of group means."""
import numpy as np, json
from scipy.stats import norm
surv, tmax = 0.7, 6
seas = np.arange(1, tmax + 1); pT = surv ** (seas - 1) * (1 - surv); pT[-1] = surv ** (tmax - 1); pT /= pT.sum()
Tbar = (seas * pT).sum()
def pever(pi): return float((pT * (1 - (1 - pi) ** seas)).sum())
def power(N, pi, d=0.2, rel=1.0, ppv=1.0):
    pe = pever(pi); n1 = N * pe; n0 = N - n1; v = 1 + (1 - rel) / rel
    se = np.sqrt(v / n1 + v / n0); return norm.cdf(ppv * d / se - 1.96)
def need(pi, d=0.2, rel=1.0, ppv=1.0):
    lo, hi = 10, 1e9
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        if power(mid, pi, d, rel, ppv) < 0.8: lo = mid
        else: hi = mid
    return hi
def sim_check(N, pi, d, rel, ppv, reps=4000, seed=1):
    rng = np.random.default_rng(seed); pe = pever(pi); v = 1 + (1 - rel) / rel; hit = 0
    for _ in range(reps):
        n1 = max(rng.binomial(int(N), pe), 2); n0 = int(N) - n1
        m1 = rng.normal(ppv * d, np.sqrt(v / n1)); m0 = rng.normal(0, np.sqrt(v / n0))
        hit += abs(m1 - m0) / np.sqrt(v / n1 + v / n0) > 1.96
    return hit / reps
if __name__ == "__main__":
    rows = []
    for pi in [0.005, 0.01, 0.02, 0.05]:
        for rel in [1.0, 0.5]:
            for ppv in [1.0, 0.6]:
                N = need(pi, 0.2, rel, ppv); rows.append(dict(pi=pi, rel=rel, ppv=ppv, males=int(np.ceil(N)), male_seasons=int(np.ceil(N * Tbar))))
    for r in rows: print(r)
    N = need(0.05, 0.2, 1.0, 1.0); print("sim check", sim_check(N, 0.05, 0.2, 1.0, 1.0), "target 0.8; Tbar", Tbar, "pever(.05)", pever(0.05))
    json.dump(dict(rows=rows, Tbar=Tbar), open("power.json", "w"))
