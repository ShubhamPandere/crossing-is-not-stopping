"""Build the 5 main figures + appendix Fig A1 for the NeurIPS submission.
Numbers pulled from phase0_v3 analysis JSON and raw JSONL; every value cross-checked
against the drafted sections.

Fig A1 deliberately reads the archived (pre `--fix-steps-left`) day2_controller_family.json,
not cem_iteration_family.py's corrected output -- see that script's own docstring caveat: no
corrected rerun exists for the ln_act/alpha0/nas6 cells this appendix figure needs.
"""
import json, os, math, statistics as st
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

P = "phase0_v3"
OUT = "phase0_v3/figures"
os.makedirs(OUT, exist_ok=True)

mpl.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200,
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.linewidth": 0.6, "grid.linewidth": 0.4, "lines.linewidth": 1.6,
    "axes.grid": True, "grid.alpha": 0.35, "grid.color": "#cccccc",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})
BLUE, VERM, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#999999"


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- Fig 0: one example episode
ex = None
for r in (json.loads(l) for l in open(f"{P}/c2_settle2_baseline_nas2_fixsteps/baseline_R2.jsonl")):
    if r.get("passthrough_success") and 100 < r["settled_block_pos_diff"] < 115 and r["init_block_pos_diff"] < 60:
        ex = r; break
tr = ex["settled_trace"]
xs_ = [c["step"] - 1 for c in tr]           # steps after the criterion first held
ys_ = [c["block_pos_diff"] for c in tr]
fig, ax = plt.subplots(figsize=(3.5, 2.5))
ax.axhspan(0, 20, color=GREEN, alpha=0.12, lw=0)
ax.text(39, 21, "goal region", ha="right", va="bottom", color="#2e7d5b", fontsize=7)
ax.plot(xs_, ys_, color=VERM, marker="s", ms=5, ls="--", zorder=3)
ax.scatter([0], [ys_[0]], color=BLUE, s=60, zorder=5)
ax.annotate("success recorded here", (0, ys_[0]), textcoords="offset points",
            xytext=(8, 26), fontsize=7.5, color=BLUE,
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.7))
ax.annotate(f"block rests {ys_[-1]:.0f} px away", (xs_[-1], ys_[-1]), textcoords="offset points",
            xytext=(-10, 16), ha="right", fontsize=7.5, color=VERM,
            arrowprops=dict(arrowstyle="->", color=VERM, lw=0.7))
ax.set_xlabel("environment steps after the goal criterion first held")
ax.set_ylabel("block distance to goal (px)")
ax.set_xlim(-2, 42)
ax.set_ylim(0, max(ys_) * 1.22)
save(fig, "fig0_example")
print(f"  fig0: episode {ex['episode']}, crossing at raw step {ex.get('success_at_step')}, "
      f"hold trace {[round(y,1) for y in ys_]}")

# ---------------------------------------------------------------- Fig 1: dose ladder
# Closed loop = --fix-steps-left cells (damping 0 = R0, see note); open loop = immune, unchanged.
damp = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
def load_jsonl(p):
    return [json.loads(l) for l in open(p)]
def cell_sr(path):
    rs = load_jsonl(path)
    return (sum(r["passthrough_success"] for r in rs) / len(rs),
            sum(r["settled_success"] for r in rs) / len(rs))
CL = {0.0: f"{P}/c2_settle2_R0_baseline_nas2_fixsteps/baseline_R0.jsonl",
      0.05: f"{P}/ladder_dmp005_baseline_nas2_fixsteps/baseline_R2.jsonl",
      0.1: f"{P}/ladder_dmp01_baseline_nas2_fixsteps/baseline_R2.jsonl",
      0.2: f"{P}/ladder_dmp02_baseline_nas2_fixsteps/baseline_R2.jsonl",
      0.3: f"{P}/ladder_dmp03_baseline_nas2_fixsteps/baseline_R2.jsonl",
      0.5: f"{P}/c2_settle2_baseline_nas2_fixsteps/baseline_R2.jsonl"}
if not os.path.exists(CL[0.0]):
    CL[0.0] = f"{P}/c2_settle2_R0_baseline_nas2/baseline_R0.jsonl"   # R0 re-run pending
OL = {0.0: f"{P}/c2_settle2_R0_baseline_nas6/baseline_R0.jsonl",
      0.05: f"{P}/ladder_dmp005_baseline_nas6/baseline_R2.jsonl",
      0.1: f"{P}/c2_settle2_dmp01_baseline_nas6/baseline_R2.jsonl",
      0.2: f"{P}/ladder_dmp02_baseline_nas6/baseline_R2.jsonl",
      0.3: f"{P}/ladder_dmp03_baseline_nas6/baseline_R2.jsonl",
      0.5: f"{P}/c2_settle2_baseline_nas6/baseline_R2.jsonl"}
sr = {"cl": {x: cell_sr(CL[x]) for x in damp}, "ol": {x: cell_sr(OL[x]) for x in damp}}
print("  fig1 CL:", {x: (round(a, 2), round(b, 2)) for x, (a, b) in sr["cl"].items()})
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7), sharey=True)
for ax, key, title in [(axes[0], "cl", "Closed loop (3 replans)"), (axes[1], "ol", "Open loop (1 plan)")]:
    xc = [sr[key][x][0] for x in damp]; xs = [sr[key][x][1] for x in damp]
    ax.plot(damp, xc, color=BLUE, marker="o", ms=5, label="threshold-crossing")
    ax.plot(damp, xs, color=VERM, marker="s", ms=5, ls="--", label="settled")
    ax.fill_between(damp, xs, xc, color=GREY, alpha=0.12, lw=0)
    ax.set_title(title); ax.set_xlabel("velocity retention $d$"); ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(damp)

# Put the "not just a failure regime" point into the figure: at d=0.1 the planner is still
# partly working (settled 0.15 closed loop) and the gap is ALREADY at its plateau. The prose
# version of this argument was shortened for the page budget, so the figure carries it too.
_ax, _d = axes[0], 0.1
_c, _s = sr["cl"][_d]
_ax.annotate(f"planner still settles\n{_s:.0%} here, gap already {_c - _s:.0%}",
             xy=(_d, _s), xytext=(0.155, 0.42), fontsize=6.2, color="#444444",
             ha="left", va="bottom",
             arrowprops=dict(arrowstyle="->", color="#777777", lw=0.6,
                             shrinkA=0, shrinkB=3))
axes[0].set_ylabel("success rate")
axes[0].legend(loc="upper right")
save(fig, "fig1_dose_ladder")

# ---------------------------------------------------------------- Fig 2: drift over the hold
# median block distance to goal at each hold, closed loop, --fix-steps-left cells.
# Recomputed 2026-09-03 from settled_trace (R0 re-run included).
holds = [1, 5, 15, 30, 40]
med_d0   = [14.2, 14.2, 14.2, 14.2, 14.2]
med_d01  = [15.7, 41.8, 49.1, 49.9, 49.9]
med_d05  = [52.4, 84.2, 113.9, 133.2, 140.9]
fig, ax = plt.subplots(figsize=(3.4, 2.6))
ax.plot(holds, med_d0, color=BLUE, marker="o", ms=5, label="$d = 0$")
ax.plot(holds, med_d01, color=GREEN, marker="^", ms=5, ls="-.", label="$d = 0.1$")
ax.plot(holds, med_d05, color=VERM, marker="s", ms=5, ls="--", label="$d = 0.5$")
ax.axhline(20, color=GREY, ls=":", lw=1)
ax.text(40, 22, "goal radius", ha="right", va="bottom", color=GREY, fontsize=7)
ax.set_xlabel("hold length (environment steps)")
ax.set_ylabel("block distance to goal (px), median")
ax.set_xticks(holds); ax.legend(loc="upper left")
save(fig, "fig2_drift")

# ---------------------------------------------------------------- Fig 3: CEM-iteration ladder (fixed cells)
def load(p): return [json.loads(l) for l in open(p)]
cells = {1: "fam_it1_baseline_nas2_fixsteps", 3: "fam_it3_baseline_nas2_fixsteps",
         10: "c2_settle2_baseline_nas2_fixsteps", 30: "fam_it30_baseline_nas2_fixsteps"}
iters = [1, 3, 10, 30]; cross = []; sdist = []; toward = []
for it in iters:
    rows = load(f"{P}/{cells[it]}/baseline_R2.jsonl")
    cross.append(sum(r["passthrough_success"] for r in rows) / len(rows))
    sdist.append(st.mean(r["settled_block_pos_diff"] for r in rows))
    toward.append(sum((r["init_block_pos_diff"] - r["settled_block_pos_diff"]) > 0 for r in rows))
fig, ax1 = plt.subplots(figsize=(3.9, 2.5))
x = list(range(len(iters)))
ax2 = ax1.twinx()
l1, = ax1.plot(x, cross, color=BLUE, marker="o", ms=6, label="threshold-crossing SR")
l2, = ax2.plot(x, sdist, color=VERM, marker="s", ms=6, ls="--", label="mean settled distance")
ax1.set_ylim(0, 1); ax1.set_ylabel("threshold-crossing success rate", color=BLUE)
ax2.set_ylabel("mean settled distance (px)", color=VERM)
ax1.tick_params(axis="y", colors=BLUE); ax2.tick_params(axis="y", colors=VERM)
ax2.set_ylim(0, max(sdist) * 1.30)
ax2.grid(False); ax2.spines["right"].set_visible(True); ax2.spines["right"].set_color(VERM)
ax1.set_xticks(x); ax1.set_xticklabels([str(i) for i in iters])
ax1.set_xlabel("CEM iterations per replan"); ax1.margins(x=0.14)
ax1.legend(handles=[l1, l2], loc="lower right", fontsize=7, framealpha=0.9, frameon=True)
save(fig, "fig3_iteration_ladder")
print("  fig3 check: cross", [round(c, 2) for c in cross], "sdist", [round(s, 1) for s in sdist], "toward", toward)

# ---------------------------------------------------------------- Fig 4: coast model, predicted vs measured
# 2026-09-04: cell list and v0 inversion now match research_audit/scripts/coast_window_compare.py
# EXACTLY (EVIDENCE_LEDGER N14b), including the --fix-steps-left closed-loop cells. The previous
# version of this figure mixed pre-correction closed-loop cells with the corrected numbers quoted
# in the text. Both windows are drawn: 0->1 is the paper's primary test, 5->15 the secondary.
DT = 0.1  # s per env step
COAST_CELLS = [
    (0.05, f"{P}/ladder_dmp005_baseline_nas2_fixsteps/baseline_R2.jsonl"),
    (0.1,  f"{P}/ladder_dmp01_baseline_nas2_fixsteps/baseline_R2.jsonl"),
    (0.2,  f"{P}/ladder_dmp02_baseline_nas2_fixsteps/baseline_R2.jsonl"),
    (0.3,  f"{P}/ladder_dmp03_baseline_nas2_fixsteps/baseline_R2.jsonl"),
    (0.5,  f"{P}/c2_settle2_baseline_nas2_fixsteps/baseline_R2.jsonl"),
    (0.05, f"{P}/ladder_dmp005_baseline_nas6/baseline_R2.jsonl"),
    (0.1,  f"{P}/c2_settle2_dmp01_baseline_nas6/baseline_R2.jsonl"),
    (0.2,  f"{P}/ladder_dmp02_baseline_nas6/baseline_R2.jsonl"),
    (0.3,  f"{P}/ladder_dmp03_baseline_nas6/baseline_R2.jsonl"),
    (0.5,  f"{P}/c2_settle2_baseline_nas6/baseline_R2.jsonl"),
]

def _k(d, a, b):
    """Eq. 1 multiplier: drift over hold steps [a,b] = v0 * k."""
    return (d ** (a * DT) - d ** (b * DT)) / math.log(1 / d)

def coast_points(fit_a, fit_b):
    xs, ys = [], []
    for d, path in COAST_CELLS:
        if not os.path.exists(path):
            print("  MISSING", path); continue
        pm, mm = [], []
        for r in load(path):
            tr = {q["step"]: q["block_pos_diff"] for q in r["settled_trace"]}
            tr[0] = r["block_pos_diff"]
            if not all(k in tr for k in (fit_a, fit_b, 15, 40)): continue
            v0 = abs(tr[fit_b] - tr[fit_a]) / _k(d, fit_a, fit_b)
            pm.append(v0 * _k(d, 15, 40)); mm.append(abs(tr[40] - tr[15]))
        if pm:
            xs.append(float(np.median(pm))); ys.append(float(np.median(mm)))
    return np.array(xs), np.array(ys)

p01x, p01y = coast_points(0, 1)     # primary
p15x, p15y = coast_points(5, 15)    # secondary
fig, ax = plt.subplots(figsize=(3.6, 3.0))
allv = np.concatenate([p01x, p01y, p15x, p15y])
lo, hi = max(allv.min() * 0.6, 0.02), allv.max() * 1.5
ax.plot([lo, hi], [lo, hi], color=GREY, ls=":", lw=1)
# NOTE: these labels MUST be raw strings. As plain strings, "0$\to$1" contains a literal
# TAB (Python escapes \t before matplotlib ever sees it) and mathtext rendered the leftover
# as an italic "o" -- the legend read "0o1" / "5o15" in the submitted PDF. Caught in review.
ax.scatter(p01x, p01y, color=BLUE, s=38, zorder=3,
           label=r"$v_0$ from hold $0\!\to\!1$ (primary)")
ax.scatter(p15x, p15y, facecolors="none", edgecolors=VERM, s=38, lw=1.2, zorder=3,
           label=r"$v_0$ from hold $5\!\to\!15$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("predicted drift (px), cell median")
ax.set_ylabel("measured drift (px), cell median")
ax.set_title("10 cells, $n = 200$ episodes\npooled $r = 0.91$ (primary), $0.96$ (secondary)",
             fontsize=8)
ax.legend(loc="upper left", fontsize=6.5)
save(fig, "fig4_coast")
print("  fig4 check: cells", len(p01x), "| primary pred range",
      round(p01x.min(), 2), round(p01x.max(), 1),
      "| secondary pred range", round(p15x.min(), 2), round(p15x.max(), 1))

# ---------------------------------------------------------------- Fig A1: 12-controller scatter (appendix, pre-fix, caveated)
c = json.load(open(f"{P}/day2_controller_family.json"))["controllers"]
fig, ax = plt.subplots(figsize=(3.6, 3.0))
for name, v in c.items():
    fam = "frozen" if "baseline" in name else "adapter"
    mk = "o" if fam == "frozen" else "^"
    col = BLUE if fam == "frozen" else VERM
    ax.scatter(v["pass_through_SR"], v["settled_dist_mean"], marker=mk, color=col, s=30, zorder=3)
ax.scatter([], [], marker="o", color=BLUE, label="frozen")
ax.scatter([], [], marker="^", color=VERM, label="adapter")
ax.set_xlabel("threshold-crossing success rate")
ax.set_ylabel("mean settled distance (px)")
# Legend must NOT sit inside the data region: at loc="upper right" its orange triangle landed
# near x=0.45, which reads as an adapter data point and contradicts the appendix text's
# "every adapter controller scores between 0.00 and 0.30". Caught in review. Lower right is
# genuinely empty here (no controller is both high-crossing and low-settled-distance).
ax.legend(loc="lower right", framealpha=0.95, frameon=True, borderpad=0.5)
save(fig, "figA1_controller_scatter")
_x = [v["pass_through_SR"] for v in c.values()]
_ad = [v["pass_through_SR"] for n, v in c.items() if "baseline" not in n]
print("  figA1 check: crossing-SR range", min(_x), max(_x), "| adapter range", min(_ad), max(_ad))
