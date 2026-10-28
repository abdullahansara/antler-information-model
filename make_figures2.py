"""Revised figures for the antler paper (Figures 1-5). Reads grid2.json (same folder) (analytic quadrature, no smoothing).
Writes Fig{1..5}_*.png (300 dpi), .tiff (600 dpi), .pdf, .eps into this folder."""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.colors import Normalize, LinearSegmentedColormap

HERE = pathlib.Path(__file__).resolve().parent
G = json.load(open(HERE / "grid2.json"))
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": True,
                     "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold"})
INK, BLUE, RED, DARKRED, GRAY, LIGHT, PALE, PALEBLUE = "#222222", "#2b6cb0", "#c8412b", "#a8331f", "#6b6b66", "#f0efec", "#dcdcd7", "#b9c9dc"


def save(fig, stem):
    fig.savefig(HERE / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(HERE / f"{stem}.tiff", dpi=600, bbox_inches="tight", pad_inches=0.03, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(HERE / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(HERE / f"{stem}.eps", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


# ============================================================ Figure 1: annual cycle with separate evidence tracks
def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.set_xlim(0, 12); ax.set_ylim(0, 8.3); ax.set_yticks([])
    ax.set_xticks([i + 0.5 for i in range(12)]); ax.set_xticklabels("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(), fontsize=8)
    ax.tick_params(length=0)
    for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
    for m in range(1, 12): ax.axvline(m, color="#e6e6e2", lw=0.5, zorder=0)
    rows = [("Antler", 7.2), ("(a) Living target\ntissue available", 5.9), ("(b) Intensity of\nagonistic contact", 4.5),
            ("(c) Lesion to living\ntissue per interaction", 3.1), ("(d) Damage recorded\nin the source studies", 1.7)]
    for t, y in rows: ax.text(-0.15, y + 0.3, t, ha="right", va="center", fontsize=7.5)

    def bar(x0, x1, y, c, txt=None, h=0.6, tc="white", hatch=None, ec="white", ls="-", lw=0.8):
        ax.add_patch(Rectangle((x0, y), x1 - x0, h, fc=c, ec=ec, lw=lw, ls=ls, hatch=hatch, zorder=2))
        if txt: ax.text((x0 + x1) / 2, y + h / 2, txt, ha="center", va="center", fontsize=7, color=tc, zorder=3)

    bar(0, 2.9, 7.2, GRAY, "hard antler"); bar(2.9, 3.2, 7.2, RED); bar(3.2, 7.7, 7.2, BLUE, "velvet growth (living)")
    bar(7.7, 12, 7.2, GRAY, "mineralized, dead antler")
    bar(0, 2.9, 5.9, PALE, "pedicle only", tc=INK); bar(2.9, 7.7, 5.9, BLUE, "pedicle and growing tip"); bar(7.7, 12, 5.9, PALE, "pedicle only; beam is dead", tc=INK)
    # (b) intensity: qualitative heights, labelled as schematic
    xs = np.linspace(0, 12, 241)
    inten = 0.18 + 0.12 * np.exp(-((xs - 5.2) / 1.6) ** 2) + 0.85 * np.exp(-((xs - 9.7) / 0.8) ** 2) + 0.2 * np.exp(-((xs - 11.2) / 0.5) ** 2)
    ax.fill_between(xs, 4.5, 4.5 + 0.8 * inten, color=PALEBLUE, ec=BLUE, lw=0.8, zorder=2)
    ax.text(5.2, 4.52 + 0.8 * 0.36 + 0.12, "velvet-period rank contacts", fontsize=7, ha="center", color=INK)
    ax.text(9.7, 4.5 + 0.8 * 1.0 + 0.1, "rut", fontsize=7, ha="center", color=INK)
    ax.text(0.25, 4.5 + 0.8 * 0.18 + 0.1, "schematic: heights are qualitative", fontsize=6.5, ha="left", color=GRAY, style="italic")
    # (c) lesion probability per interaction: unmeasured everywhere
    bar(3.2, 7.7, 3.1, "white", None, tc=DARKRED, hatch="////", ec=DARKRED, ls=(0, (3, 2)), lw=1.0)
    ax.text(5.45, 3.4, "not measured in free-living animals", ha="center", va="center", fontsize=7, color=DARKRED, zorder=4, bbox=dict(fc="white", ec="none", pad=1.5))
    bar(8.7, 10.9, 3.1, "white", "no living tissue exposed", tc=GRAY, ec=GRAY, ls=(0, (3, 2)), lw=0.8)
    # (d) documented
    bar(8.4, 11.1, 1.7, RED, "hard antler, eye, leg (wild)")
    bar(3.2, 6.6, 1.7, "white", "surgery, accidents (captive)", tc=BLUE, ec=BLUE, ls=(0, (3, 2)), lw=1.0)
    ax.text(12.0, 0.98, "solid = recorded in free-living or rutting animals;  dashed = experimental or captive reports;  hatched = no measurement exists.\n"
            "Timings are approximate and synthesised from the cited studies; no row shows measured seasonal rates.", fontsize=6.5, ha="right", va="top", color=INK)
    save(fig, "Fig1_annual_cycle")


# ============================================================ Figure 2: causal structure with four evidence conventions
def fig2():
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.axis("off"); ax.set_xlim(0, 104); ax.set_ylim(-9, 80)
    ax.add_patch(Rectangle((0, 0), 51, 74, fc="#f7f6f3", ec="none", zorder=0)); ax.add_patch(Rectangle((51, 0), 53, 74, fc="#eef2f7", ec="none", zorder=0))
    ax.text(25, 76.5, "Cycle t (velvet growth, rut)", ha="center", fontsize=8, fontweight="bold"); ax.text(77, 76.5, "Cycle t+1 (assessment by receivers)", ha="center", fontsize=8, fontweight="bold")
    N = {"Q": (12, 40, "Competitive\nquality (latent)"), "F": (38, 62, "Contests"), "L": (38, 46, "Lesion to living tissue\n(pedicle, growing tip)"),
         "AGE": (38, 28, "Age, nutrition,\ncondition"), "K": (12, 8, "Rank during\nvelvet period"), "H": (38, 8, "Rank-modulated\ngrowth (endocrine)"),
         "SV": (64, 62, "Survival"), "R": (64, 46, "Carry-over branch\nat the lesion site"), "S": (64, 28, "Antler size and\nbaseline branching"),
         "D": (90, 46, "Observed form,\ndiagnosed origin"), "RR": (90, 20, "Receiver response:\nchoice, assessment")}
    for k, (x, y, t) in N.items():
        ax.add_patch(FancyBboxPatch((x - 10.2, y - 5), 20.4, 10, boxstyle="round,pad=0.3,rounding_size=1.2", fc="white", ec=INK, lw=0.8, zorder=2))
        ax.text(x, y, t, ha="center", va="center", fontsize=7.2, color=INK, zorder=3)
    STY = {"exp": dict(c=BLUE, lw=1.6, ls="-", ah="-|>"), "assoc": dict(c="#555555", lw=1.0, ls="-", ah="->"),
           "hyp": dict(c=DARKRED, lw=1.2, ls=(0, (4, 2)), ah="->")}

    def A(p1, p2, kind, rad=0.0, lag=False, q=None):
        s = STY[kind]
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=s["ah"], mutation_scale=9, lw=s["lw"], color=s["c"], ls=s["ls"], connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1, zorder=4))
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        if lag: ax.text(mx, my + 2.0, "lag", fontsize=7, style="italic", ha="center", color=s["c"], bbox=dict(fc="white", ec="none", pad=0.4), zorder=5)
        if kind == "hyp" and q: ax.text(*q, "?", fontsize=9, fontweight="bold", color=DARKRED, ha="center", va="center", zorder=5, bbox=dict(fc="white", ec="none", pad=0.2))

    A((21, 44.8), (27.8, 60), "hyp", q=(22.5, 54))                # quality -> contests (kappa unmeasured)
    A((38, 56.8), (38, 51.6), "hyp", q=(40.3, 54.2))              # contests -> lesion
    A((48.4, 46), (53.6, 46), "exp", lag=True)                    # lesion -> carry-over (captive experiments)
    A((46.6, 50.8), (55.2, 57.6), "hyp", q=(52.2, 55.6))          # lesion -> survival
    A((74.4, 60), (88, 51.6), "hyp", q=(83, 57.5))
    A((74.4, 46), (79.6, 46), "hyp", q=(77, 48.5))                # carry-over -> diagnosed origin
    A((22.4, 38), (53.8, 31.5), "assoc")               # quality -> size
    A((48.4, 28), (53.6, 28), "assoc")                            # age -> size
    A((74.4, 30), (80, 24.5), "exp")                               # size -> receiver response (experiment on size only)
    A((90, 40.8), (90, 25.6), "hyp", q=(92.6, 33))                # diagnosed origin -> receiver response
    A((12, 34.6), (12, 13.6), "assoc")                             # quality -> rank
    A((22.4, 8), (27.8, 8), "assoc")                               # rank -> growth
    A((46, 11), (58, 22.8), "assoc", rad=-0.1)                     # growth -> size/branching
    ax.text(75, 6.5, "Experiments on receiver response used antler size only;\nno study has tested injury-origin form.", fontsize=7, color=INK, ha="center", va="center", style="italic")
    # legend
    y0 = -2
    for i, (kind, lab) in enumerate([("exp", "link supported by isolated experimental or accidental observations (weak evidence)"),
                                      ("assoc", "observed association; causation not established"),
                                      ("hyp", "hypothetical, unquantified or untested; ?")]):
        yy = y0 - 2.6 * i; s = STY[kind]
        ax.add_patch(FancyArrowPatch((1, yy), (9, yy), arrowstyle=s["ah"], mutation_scale=9, lw=s["lw"], color=s["c"], ls=s["ls"]))
        ax.text(11, yy, lab, va="center", fontsize=7.2)
    ax.text(70, y0, "shading and 'lag' mark effects that cross the annual cast", fontsize=7.2, va="center")
    save(fig, "Fig2_causal_diagram")


# ============================================================ Figure 3: information maps (unsmoothed) with prevalence and PPV
def fig3():
    ks, gs = np.array(G["kappa"]), np.array(G["gamma"]); S = G["scen"]
    cm = LinearSegmentedColormap.from_list("div", ["#2166ac", "#f0efec", "#b2182b"])
    D = lambda n: np.array(S[n]["fine"]["delta"])
    panels = [("(a) Baseline", D("A baseline")), ("(b) Costly injury (c = 2.5)", D("B costly injury (c=2.5)")),
              ("(c) Size tied to quality (r = 0.8)", D("D size tightly tied to Q (r=.8)")), ("(d) Poor diagnosis (0.6, 0.15)", D("C poor diagnosis (se=.6, fp=.15)"))]
    m = 0.6; norm = Normalize(-m, m)
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 5.4), constrained_layout=True, sharex=True, sharey=True)
    for a, (t, Z) in zip([axs[0, 0], axs[0, 1], axs[0, 2], axs[1, 0]], panels):
        im = a.pcolormesh(ks, gs, Z, cmap=cm, norm=norm, shading="nearest", rasterized=True)
        a.contour(ks, gs, Z, levels=[0], colors="white", linewidths=2.8); a.contour(ks, gs, Z, levels=[0], colors="black", linewidths=1.4)
        cs = a.contour(ks, gs, Z, levels=[0.1, 0.2, 0.3, 0.4, 0.5], colors="#222222", linewidths=0.5)
        for tx in a.clabel(cs, fmt="%.1f", fontsize=6.5, inline=True, inline_spacing=3): tx.set_path_effects([pe.withStroke(linewidth=1.6, foreground="white")])
        a.set_title(t, loc="left", fontsize=7.2); a.axvline(0, color="#888888", lw=0.5, ls=":")
    g = np.linspace(0, 2, 50); axs[0, 0].plot(0.25 * g, g, color="black", ls=(0, (4, 2)), lw=0.9)
    axs[0, 0].text(-0.46, 1.86, "dashed: " + r"$\kappa=0.25\gamma$" + "\n(exposure only)", fontsize=6.5, va="center", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    cb = fig.colorbar(im, ax=[axs[0, 0], axs[0, 1], axs[0, 2], axs[1, 0]], shrink=0.7, pad=0.01, aspect=22, location="bottom")
    cb.set_label(r"$\Delta$ (SD of quality, diagnosed vs undiagnosed males of equal size)", fontsize=7); cb.ax.tick_params(labelsize=7)
    base = S["A baseline"]["fine"]
    for a, key, t, cmap, vmin, vmax in [(axs[1, 1], "prev", "(e) Diagnosed share of survivors", "Greys", 0, 0.12), (axs[1, 2], "ppv", "(f) Positive predictive value", "Greens", 0.3, 0.8)]:
        Z = np.array(base[key]); imz = a.pcolormesh(ks, gs, Z, cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest", rasterized=True)
        cz = a.contour(ks, gs, Z, levels=[0.05, 0.07, 0.09] if key == "prev" else [0.6, 0.7], colors="black", linewidths=0.5)
        for tx in a.clabel(cz, fmt="%.2f", fontsize=6.5, inline=True): tx.set_path_effects([pe.withStroke(linewidth=1.6, foreground="white")])
        a.set_title(t, loc="left", fontsize=7.2); fig.colorbar(imz, ax=a, shrink=0.7, aspect=18, pad=0.02, location="right").ax.tick_params(labelsize=6.5)
    for a in axs[1]: a.set_xlabel(r"Exposure gradient $\kappa$", fontsize=7.5)
    for a in axs[:, 0]: a.set_ylabel(r"Victory gradient $\gamma$", fontsize=7.5)
    for a in axs.ravel(): a.tick_params(labelsize=7)
    save(fig, "Fig3_parameter_space")


# ============================================================ Figure 4: rarity
def fig4():
    R = G["rarity"]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9), constrained_layout=True, sharex=True)
    sty = {"baseline": (BLUE, "-", "baseline (sens. 0.9, false-pos. 0.02)"), "poor": ("#d98c1f", "-", "poor (0.6, 0.15)"), "near-chance": (DARKRED, "--", "near chance (0.5, 0.4)")}
    for dl, (c, ls, lab) in sty.items():
        rows = R[f"kappa=0.5,gamma=1|{dl}"]; psi = [r["psi_l"] for r in rows]
        axs[0].plot(psi, [r["prev"] for r in rows], color=c, ls=ls, lw=1.4, label=lab); axs[1].plot(psi, [r["ppv"] for r in rows], color=c, ls=ls, lw=1.4)
        axs[2].plot(psi, [r["delta"] for r in rows], color=c, ls=ls, lw=1.4)
    rows = R["kappa=0,gamma=2|baseline"]; axs[2].plot([r["psi_l"] for r in rows], [r["delta"] for r in rows], color=BLUE, ls=":", lw=1.2)
    axs[2].text(2.5e-4, -0.07, r"$\kappa=0,\gamma=2$ (baseline)", fontsize=6.5, color=BLUE)
    axs[2].text(2.5e-4, 0.17, r"$\kappa=0.5,\gamma=1$", fontsize=6.5, color=BLUE)
    for a, t, yl in zip(axs, ["(a) Diagnosed share of survivors", "(b) Positive predictive value", r"(c) $\Delta$"], ["share", "P(injury-origin | diagnosed)", "SD of quality"]):
        a.set_xscale("log"); a.set_title(t, loc="left", fontsize=7.5); a.set_ylabel(yl, fontsize=7.5); a.set_xlabel(r"Injury probability per lost contest, $\psi_l$ ($\psi_w=\psi_l/3$)", fontsize=7); a.tick_params(labelsize=7)
        a.axvline(0.06, color="#999999", lw=0.6, ls=":"); a.set_xlim(1e-5, 0.1)
    axs[0].legend(fontsize=6.3, frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.55)); axs[2].axhline(0, color="black", lw=0.5)
    axs[0].text(0.058, 0.44, "baseline\nvalue", fontsize=6, ha="right", va="center", color="#666666")
    save(fig, "Fig4_rarity")


# ============================================================ Figure 5: prospective identification design
def fig5():
    fig, ax = plt.subplots(figsize=(7.2, 3.6)); ax.axis("off"); ax.set_xlim(0, 104); ax.set_ylim(0, 56)
    cols = [("Cycle t", 2, 33), ("Cycle t+1", 36, 67), ("Cycle t+2", 70, 101)]
    for t, x0, x1 in cols:
        ax.add_patch(Rectangle((x0, 4), x1 - x0, 44, fc="#f4f3f0", ec="none")); ax.text((x0 + x1) / 2, 50.5, t, ha="center", fontsize=8.5, fontweight="bold")

    def box(x, y, w, h, txt, fc="white", ec=INK, ls="-", fs=7):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.8", fc=fc, ec=ec, lw=0.9, ls=ls)); ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs)

    def arr(p1, p2, c=INK, ls="-", rad=0.0):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=8, lw=1.0, color=c, ls=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
    box(4, 36, 27, 9, "Velvet period: marked males\nfilmed; interactions counted;\nlesions scored blind", fc="#e7eef7")
    box(4, 22, 27, 9, "Lesion site and severity\nrecorded (photo, 3D scan,\nveterinary check if possible)", fc="#e7eef7")
    box(4, 8, 27, 9, "Measured at t: age, nutrition,\ntestosterone, size, rank,\ncontest record", fc="white")
    box(38, 36, 27, 9, "Cast antler scanned in 3D;\nbranch topology compared with\nsame animal's pre-lesion record", fc="#e7eef7")
    box(38, 22, 27, 9, "Contralateral antler and\nunwounded matched regions\nas within-animal controls", fc="white")
    box(38, 8, 27, 9, "Blind raters and classifier\nassign origin of branches;\nsensitivity, specificity, PPV", fc="white")
    box(72, 36, 27, 9, "Does the site-specific branch\nfade, persist or recur?\n(lag structure)", fc="#e7eef7")
    box(72, 22, 27, 9, "Opponent-adjusted contest\nsuccess and survival;\nresidual value given size", fc="white")
    box(72, 8, 27, 9, "Only if prevalence is non-trivial:\nrandomised stimuli and paternity", fc="white", ec=DARKRED, ls=(0, (3, 2)))
    arr((17.5, 35.5), (17.5, 31.6)); arr((31.5, 40.5), (37.5, 40.5)); arr((65.5, 40.5), (71.5, 40.5)); arr((31.5, 26.5), (37.5, 26.5)); arr((65.5, 12.5), (71.5, 12.5)); arr((51.5, 21.6), (51.5, 17.6))
    arr((65.5, 26.5), (71.5, 26.5)); arr((31.5, 12.5), (37.5, 12.5), c=GRAY)
    ax.text(52, 1.5, "Blue boxes: the cheapest decisive measurements (lesion rate, then site-specific carry-over). Dashed box: conditional on those results.", ha="center", fontsize=6.8)
    save(fig, "Fig5_prospective_design")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); print("ok")
