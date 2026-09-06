"""Per-cell inference for the dose-ladder table (Table 1 / tab:ladder), plus a
monotonic-trend test on the crossing-vs-settled gap across the retention ladder.
Requested by the simulated peer-review panel, item M1 (CRITICAL): the ladder table
reports bare proportions with no uncertainty. This computes, from the raw JSONL used
by figures/make_figs.py (fig1_dose_ladder), for every (d, cadence) cell:
  - exact paired McNemar test, crossing_success vs settled_success (same 20 episodes)
  - a paired-bootstrap 95% CI on the gap (crossing_SR - settled_SR), 20000 resamples,
    consistent with the paper's existing stats convention (02_setup.tex)
and, pooled across cells, a monotonic-trend test: logistic regression of settled
success on d, restricted to episodes that threshold-crossed (i.e. among crossings,
does P(settle | crossed) fall as d rises).
"""
import json
from pathlib import Path as _Path
import numpy as np
from scipy.stats import binomtest
import statsmodels.api as sm

P = str(_Path(__file__).resolve().parent.parent.parent / "phase0_v3")
DAMP = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
RNG = np.random.default_rng(0)

CL = {0.0: f"{P}/c2_settle2_R0_baseline_nas2_fixsteps/baseline_R0.jsonl",
      0.05: f"{P}/ladder_dmp005_baseline_nas2_fixsteps/baseline_R2.jsonl",
      0.1: f"{P}/ladder_dmp01_baseline_nas2_fixsteps/baseline_R2.jsonl",
      0.2: f"{P}/ladder_dmp02_baseline_nas2_fixsteps/baseline_R2.jsonl",
      0.3: f"{P}/ladder_dmp03_baseline_nas2_fixsteps/baseline_R2.jsonl",
      0.5: f"{P}/c2_settle2_baseline_nas2_fixsteps/baseline_R2.jsonl"}
OL = {0.0: f"{P}/c2_settle2_R0_baseline_nas6/baseline_R0.jsonl",
      0.05: f"{P}/ladder_dmp005_baseline_nas6/baseline_R2.jsonl",
      0.1: f"{P}/c2_settle2_dmp01_baseline_nas6/baseline_R2.jsonl",
      0.2: f"{P}/ladder_dmp02_baseline_nas6/baseline_R2.jsonl",
      0.3: f"{P}/ladder_dmp03_baseline_nas6/baseline_R2.jsonl",
      0.5: f"{P}/c2_settle2_baseline_nas6/baseline_R2.jsonl"}


def load(p):
    return [json.loads(l) for l in open(p)]


def mcnemar_exact(cross, settled):
    """Exact two-sided McNemar on paired binary arrays. b = crossed&!settled,
    c = settled&!crossed (should be ~0: settling implies having crossed)."""
    cross = np.asarray(cross, dtype=bool)
    settled = np.asarray(settled, dtype=bool)
    b = int(np.sum(cross & ~settled))
    c = int(np.sum(~cross & settled))
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue
    return b, c, p


def paired_bootstrap_gap_ci(cross, settled, n_resamples=20000):
    cross = np.asarray(cross, dtype=float)
    settled = np.asarray(settled, dtype=float)
    n = len(cross)
    idx = RNG.integers(0, n, size=(n_resamples, n))
    gaps = cross[idx].mean(axis=1) - settled[idx].mean(axis=1)
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return float(lo), float(hi)


print(f"{'d':>5} {'cadence':>8} {'n':>3} {'cross':>6} {'settled':>7} {'gap(pp)':>8} "
      f"{'gap 95% CI':>16} {'b':>3} {'c':>3} {'McNemar p':>10}")
results = {}
pooled_settled_given_crossed = {"d": [], "settled": []}
for label, cells in [("closed", CL), ("open", OL)]:
    for d in DAMP:
        rows = load(cells[d])
        assert len(rows) == 20, (label, d, len(rows))
        cross = [r["passthrough_success"] for r in rows]
        settled = [r["settled_success"] for r in rows]
        cross_sr = np.mean(cross)
        settled_sr = np.mean(settled)
        gap = 100 * (cross_sr - settled_sr)
        b, c, p = mcnemar_exact(cross, settled)
        lo, hi = paired_bootstrap_gap_ci(cross, settled)
        results[(label, d)] = dict(cross_sr=cross_sr, settled_sr=settled_sr, gap=gap,
                                    b=b, c=c, p=p, ci=(100 * lo, 100 * hi))
        print(f"{d:5.2f} {label:>8} {len(rows):3d} {cross_sr:6.2f} {settled_sr:7.2f} "
              f"{gap:8.1f} [{100*lo:6.1f},{100*hi:5.1f}] {b:3d} {c:3d} {p:10.4g}")
        for r in rows:
            if r["passthrough_success"]:
                pooled_settled_given_crossed["d"].append(d)
                pooled_settled_given_crossed["settled"].append(int(r["settled_success"]))

print("\n--- pooled trend test: P(settled | crossed) as a function of d ---")
dv = np.array(pooled_settled_given_crossed["d"])
sv = np.array(pooled_settled_given_crossed["settled"])
print(f"n crossed episodes pooled (both cadences, all 6 rungs) = {len(dv)}")
X = sm.add_constant(dv)
model = sm.Logit(sv, X).fit(disp=0)
print(model.summary2().tables[1])
slope, se = model.params[1], model.bse[1]
z = slope / se
from scipy.stats import norm
p_trend = 2 * (1 - norm.cdf(abs(z)))
print(f"slope on d = {slope:.4f} (se {se:.4f}), z = {z:.3f}, two-sided p = {p_trend:.3g}")

print("\n--- excluding d=0 (no variance there) ---")
mask = dv > 0
X2 = sm.add_constant(dv[mask])
model2 = sm.Logit(sv[mask], X2).fit(disp=0)
slope2, se2 = model2.params[1], model2.bse[1]
z2 = slope2 / se2
p_trend2 = 2 * (1 - norm.cdf(abs(z2)))
print(f"slope on d = {slope2:.4f} (se {se2:.4f}), z = {z2:.3f}, two-sided p = {p_trend2:.3g}")
