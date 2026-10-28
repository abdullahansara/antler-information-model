# -*- coding: utf-8 -*-
"""Combat-history cue model, revision 2.

Generative model for one male and one cycle (t -> t+1):
  Q ~ N(0,1)                          latent competitive quality (the analyst's variable, never seen by receivers)
  S = r Q + sqrt(1-r^2) e             observed size score (receivers condition on it)
  F | Q ~ Poisson(lam exp(kappa Q))   contests entered; contests independent given Q
  win | Q = s(Q)                      s(Q) = logistic(gamma Q) (opponent fixed at the mean) or an average over opponents
  injury to regenerative tissue per contest: psi_l (loss) or psi_w (win)  -> number of injuries is a thinned Poisson process
  I = 1 if at least one injury; R = 1 if the injury leaves an injury-origin branch next cycle (prob rho given I)
  A = survival to the assessment: logistic(a0 + bq Q - c I)      (receivers see survivors only)
  D = diagnosis of 'injury-origin branch': prob se if R = 1, prob fp otherwise (look-alikes)
Because contests are independent given Q, wins ~ Poisson(m s) and losses ~ Poisson(m (1-s)) are independent, so
  P(I = 1 | Q) = 1 - exp(-mu(Q)),  mu(Q) = m(Q) [psi_l (1-s) + psi_w s]
and every quantity below is an integral over Q alone (and S for the size-matched contrasts). We evaluate the integrals by
quadrature, which removes Monte Carlo error, and check them against a direct simulation (check_mc() in this file).

Reported for survivors:  delta_ols (coefficient on D in the regression of Q on D and S),
  delta_strat (size-matched contrast E[Q|D=1,S,A=1] - E[Q|D=0,S,A=1] averaged over the diagnosed males' sizes),
  prev (P(D=1|A=1)), ppv (P(R=1|D=1,A=1)), mi (I(Q;D|S,A=1) in bits).
"""
import numpy as np

SEED = 7
BASE = dict(lam=3.0, psi_l=0.06, psi_w=0.02, rho=0.5, c=1.0, bq=0.6, a0=1.5, r=0.5, se=0.9, fp=0.02, opp=None, psi_q=0.0)
sig = lambda x: 1 / (1 + np.exp(-x))
XG = np.linspace(-6.5, 6.5, 521)
PHI = np.exp(-XG ** 2 / 2) / np.sqrt(2 * np.pi) * (XG[1] - XG[0])


def win_prob(x, gamma, opp):
    if opp is None:                       # opponent fixed at the population mean
        return sig(gamma * x)
    rho_o = opp                           # opponent quality correlated rho_o with own quality (assortative contests)
    z = np.linspace(-5, 5, 81); wz = np.exp(-z ** 2 / 2); wz /= wz.sum()
    oq = rho_o * x[:, None] + np.sqrt(1 - rho_o ** 2) * z[None, :]
    return (sig(gamma * (x[:, None] - oq)) * wz[None, :]).sum(1)


def parts(kappa, gamma, **kw):
    p = dict(BASE); p.update(kw)
    x = XG
    s = win_prob(x, gamma, p["opp"])
    m = p["lam"] * np.exp(kappa * x)
    pl = p["psi_l"] * np.exp(-p["psi_q"] * x); pw = p["psi_w"] * np.exp(-p["psi_q"] * x)   # psi_q: quality protects independent of victories
    mu = m * (pl * (1 - s) + pw * s)
    pI = 1 - np.exp(-mu)
    a1 = sig(p["a0"] + p["bq"] * x - p["c"]); a0 = sig(p["a0"] + p["bq"] * x)
    pa = pI * a1 + (1 - pI) * a0
    pR = pI * p["rho"] * a1
    pD = p["se"] * pR + p["fp"] * (pa - pR)
    return p, x, pI, pa, pR, pD


def metrics(kappa, gamma, full=False, **kw):
    p, x, pI, pa, pR, pD = parts(kappa, gamma, **kw)
    A = (PHI * pa).sum()
    E = lambda g: (PHI * pa * g).sum() / A
    ED = (PHI * pD).sum() / A
    EQ, EQ2 = E(x), E(x ** 2)
    EDQ = (PHI * pD * x).sum() / A
    r = p["r"]
    ES, ES2, EQS, EDS = r * EQ, r * r * EQ2 + (1 - r * r), r * EQ2, r * EDQ
    M = np.array([[1, ED, ES], [ED, ED, EDS], [ES, EDS, ES2]]); y = np.array([EQ, EDQ, EQS])
    delta = np.linalg.solve(M, y)[1]
    out = dict(delta=delta, prev=ED, ppv=(PHI * p["se"] * pR).sum() / (PHI * pD).sum() if ED > 0 else np.nan, inj=(PHI * pI).sum(),
               inj_surv=(PHI * pI * 0).sum())
    if full:
        sg = np.linspace(-4.5, 4.5, 181); v = 1 - r * r
        K = np.exp(-(sg[:, None] - r * x[None, :]) ** 2 / (2 * v)) / np.sqrt(2 * np.pi * v)       # N(s; r x, v)
        wd = PHI[None, :] * K * pD[None, :]; w0 = PHI[None, :] * K * (pa - pD)[None, :]; wa = PHI[None, :] * K * pa[None, :]
        n1, n0 = wd.sum(1), w0.sum(1)
        ok = (n1 > 1e-12) & (n0 > 1e-12)
        m1 = (wd * x[None, :]).sum(1)[ok] / n1[ok]; m0 = (w0 * x[None, :]).sum(1)[ok] / n0[ok]
        out["strat"] = float((n1[ok] * (m1 - m0)).sum() / n1[ok].sum())
        h = lambda q: -(q * np.log2(np.clip(q, 1e-15, 1)) + (1 - q) * np.log2(np.clip(1 - q, 1e-15, 1)))
        pDs = wd.sum(1) / np.clip(wa.sum(1), 1e-300, None); ws = wa.sum(1); ws = ws / ws.sum()
        HDS = (ws * h(pDs)).sum()
        pDq = np.where(pa > 0, pD / np.clip(pa, 1e-300, None), 0); wq = PHI * pa / A
        HDQ = (wq * h(pDq)).sum()                      # D depends on (Q,S) only through Q
        out["mi"] = float(HDS - HDQ)
    return out


def grid(kappas, gammas, full=False, **kw):
    keys = ["delta", "prev", "ppv"] + (["strat", "mi"] if full else [])
    G = {k: np.zeros((len(gammas), len(kappas))) for k in keys}
    for i, g in enumerate(gammas):
        for j, k in enumerate(kappas):
            o = metrics(k, g, full=full, **kw)
            for kk in keys: G[kk][i, j] = o[kk]
    return G


# ---- direct Monte Carlo (second implementation) with extra mechanisms -----------------------------------------------------
def simulate(kappa, gamma, N=400000, seed=SEED, overdisp=None, age=False, fp_size=0.0, **kw):
    """overdisp: gamma-frailty shape for contest number (None = Poisson). age: age affects size, contests, look-alike branching.
    fp_size: look-alike probability rises with size score on the logit scale."""
    p = dict(BASE); p.update(kw)
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal(N)
    A_ = rng.standard_normal(N) if age else np.zeros(N)
    S = p["r"] * Q + 0.5 * A_ + np.sqrt(1 - p["r"] ** 2) * rng.standard_normal(N) * (np.sqrt(1 - 0.25) if age else 1)
    m = p["lam"] * np.exp(kappa * Q + (0.4 * A_ if age else 0))
    if overdisp: m = m * rng.gamma(overdisp, 1 / overdisp, N)
    F = rng.poisson(m)
    wins = rng.binomial(F, win_prob(Q, gamma, p["opp"]) if p["opp"] is not None else sig(gamma * Q)); loss = F - wins
    pl = p["psi_l"] * np.exp(-p["psi_q"] * Q); pw = p["psi_w"] * np.exp(-p["psi_q"] * Q)
    I = rng.random(N) > (1 - pw) ** wins * (1 - pl) ** loss
    R = I & (rng.random(N) < p["rho"])
    surv = rng.random(N) < sig(p["a0"] + p["bq"] * Q - p["c"] * I - (0.2 * A_ if age else 0))
    fp = p["fp"] * (np.exp(0.7 * A_) if age else 1.0)
    if fp_size: fp = sig(np.log(p["fp"] / (1 - p["fp"])) + fp_size * S)
    D = (R & (rng.random(N) < p["se"])) | ((~R) & (rng.random(N) < fp))
    Q, S, D, R = Q[surv], S[surv], D[surv], R[surv]
    X = np.column_stack([np.ones_like(Q), D.astype(float), S]); beta = np.linalg.lstsq(X, Q, rcond=None)[0]
    return dict(delta=beta[1], prev=D.mean(), ppv=(D & R).sum() / max(D.sum(), 1))


def check_mc(reps=5):
    """Monte Carlo errors and agreement with the quadrature at several cells."""
    rows = []
    for (k, g) in [(0, 0), (0.5, 1.0), (0, 2.0), (0.25, 2.0), (1.0, 0.5)]:
        sims = [simulate(k, g, N=400000, seed=SEED + i) for i in range(reps)]
        d = np.array([s["delta"] for s in sims]); pr = np.array([s["prev"] for s in sims]); pp = np.array([s["ppv"] for s in sims])
        a = metrics(k, g)
        rows.append(dict(kappa=k, gamma=g, quad_delta=a["delta"], mc_delta=d.mean(), mc_delta_sd=d.std(ddof=1), quad_prev=a["prev"], mc_prev=pr.mean(),
                         quad_ppv=a["ppv"], mc_ppv=pp.mean()))
    return rows


if __name__ == "__main__":
    import json, time
    t0 = time.time()
    SCEN = {"A baseline": {}, "B costly injury (c=2.5)": dict(c=2.5), "C poor diagnosis (se=.6, fp=.15)": dict(se=0.6, fp=0.15),
            "D size tightly tied to Q (r=.8)": dict(r=0.8), "E near-chance diagnosis (se=.5, fp=.4)": dict(se=0.5, fp=0.4),
            "F no distinctive morphology (se=fp=.05)": dict(se=0.05, fp=0.05)}
    ks = np.linspace(-0.5, 1.0, 61); gs = np.linspace(0, 2.0, 41)         # fine grid for the figure (kappa<0: low quality drives contests)
    kc = np.linspace(-0.5, 1.0, 16); gc = np.linspace(0, 2.0, 11)
    out = {"kappa": ks.tolist(), "gamma": gs.tolist(), "kappa_c": kc.tolist(), "gamma_c": gc.tolist(), "scen": {}}
    for name, kw in SCEN.items():
        G = grid(ks, gs, **kw); Gc = grid(kc, gc, full=True, **kw)
        out["scen"][name] = {"fine": {k: v.tolist() for k, v in G.items()}, "coarse": {k: v.tolist() for k, v in Gc.items()}}
        print(name, "delta", G["delta"].min().round(3), G["delta"].max().round(3), "prev", G["prev"].mean().round(3), "ppv", np.nanmean(G["ppv"]).round(3),
              "strat range", Gc["strat"].min().round(3), Gc["strat"].max().round(3), "MI max", Gc["mi"].max().round(5), flush=True)
    # rarity: scale both injury probabilities down by a factor, log spaced
    f = np.logspace(0, -4, 25)
    rar = {}
    for lab, (k, g) in {"kappa=0.5,gamma=1": (0.5, 1.0), "kappa=0,gamma=0": (0, 0.0), "kappa=0,gamma=2": (0, 2.0)}.items():
        for dl, kw in {"baseline": {}, "poor": dict(se=0.6, fp=0.15), "near-chance": dict(se=0.5, fp=0.4)}.items():
            rows = []
            for ff in f:
                o = metrics(k, g, full=True, psi_l=0.06 * ff, psi_w=0.02 * ff, **kw)
                rows.append(dict(f=ff, psi_l=0.06 * ff, delta=o["delta"], strat=o["strat"], prev=o["prev"], ppv=o["ppv"], mi=o["mi"]))
            rar[f"{lab}|{dl}"] = rows
    out["rarity"] = rar
    # robustness cells (kappa=0.5, gamma=1 and kappa=0, gamma=2) by quadrature or direct simulation
    rob = {}
    for (k, g) in [(0.5, 1.0), (0.0, 2.0), (0.0, 0.0)]:
        r_ = {"baseline": metrics(k, g, full=True)}
        r_["assortative opponents (rho=0.7)"] = metrics(k, g, full=True, opp=0.7)
        r_["quality protects against injury (psi_q=0.5)"] = metrics(k, g, full=True, psi_q=0.5)
        r_["no survival cost (c=0)"] = metrics(k, g, full=True, c=0.0)
        r_["overdispersed contests (shape 1)"] = simulate(k, g, overdisp=1.0)
        r_["age drives contests, size and look-alikes"] = simulate(k, g, age=True)
        r_["look-alikes rise with size (logit slope 0.7)"] = simulate(k, g, fp_size=0.7)
        rob[f"kappa={k},gamma={g}"] = {a: {kk: float(vv) for kk, vv in b.items()} for a, b in r_.items()}
    out["robust"] = rob
    out["check_mc"] = check_mc()
    json.dump(out, open("grid2.json", "w"))
    print("done", round(time.time() - t0), "s")
    for r_ in out["check_mc"]: print({k: round(v, 4) for k, v in r_.items()})
