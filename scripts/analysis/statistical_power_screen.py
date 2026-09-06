"""
day2_screen_power.py — the acceptance-screen power/discrimination table (paper §7).

Read-only: consumes archived per-episode JSONL under phase0_v3/, writes one JSON.
No production code path.

The question §7 asks is NOT "how many episodes does a paired probe need to detect a
difference" — it is "does the probe flag the adapters that are actually harmful and
leave alone the ones that are not". So the table is a 2x2: two candidate screen
statistics crossed with two adapters whose ground truth is already established on the
paper's own settle-validated metric.

  case HELPFUL  = ln_act @ damping 0.5   -> settled dist -40.2 px vs frozen (n=50,
                                            p=0.0001, ledger N12-n50). Flagging it is
                                            a FALSE ALARM.
  case HARMFUL  = ln_act @ damping 0.1   -> settled dist +16.9 px vs frozen (n=20,
                                            p=0.044, ledger B2-transfer-01). Flagging
                                            it is a CORRECT DETECTION.

Both screens are one-sided in the "chart is worse" direction, alpha = 0.05:
  pass-through McNemar  — exact, on the per-step threshold-crossing criterion
  settled-dist Wilcoxon — signed-rank, on block distance after the 40-step hold

Calibration is measured by sign-flipping each paired difference (the exact null for a
symmetric paired test), not by a frozen-vs-frozen comparison: two runs of an identical
config have near-zero discordance by construction, so their 0% flag rate reflects the
test having nothing to fire on rather than correct calibration.
"""
from __future__ import annotations

import glob
import json
import os
import random
import statistics as st
from math import comb

from scipy.stats import wilcoxon

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase0_v3")
ALPHA = 0.05
NS = (5, 10, 15, 20)
TRIALS = 4000

CASES = {
    # label: (chart_dir, frozen_dir, ground truth on the settle-validated metric)
    "HELPFUL  ln_act @ damping 0.5, nas=2": ("c2_settle2_ln_act_nas2", "c2_settle2_baseline_nas2"),
    "HELPFUL  ln_act @ damping 0.5, nas=6": ("c2_settle2_ln_act_nas6", "c2_settle2_baseline_nas6"),
    "HARMFUL  ln_act @ damping 0.1, nas=2": ("dmp01_transfer_ln_act_nas2", "c2_settle2_dmp01_baseline_nas2"),
    "HARMFUL  ln_act @ damping 0.1, nas=6": ("dmp01_transfer_ln_act_nas6", "c2_settle2_dmp01_baseline_nas6"),
}


def _load(d):
    f = glob.glob(os.path.join(ROOT, d, "*.jsonl"))
    if not f:
        raise FileNotFoundError(d)
    return {json.loads(l)["episode"]: json.loads(l) for l in open(f[0]) if l.strip()}


def load_pairs(chart_dir, frozen_dir):
    """(chart_passed, frozen_passed, chart_settled_dist, frozen_settled_dist) per task."""
    A, B = _load(chart_dir), _load(frozen_dir)
    ks = sorted(set(A) & set(B))
    mism = sum(1 for k in ks if abs(A[k]["init_block_pos_diff"] - B[k]["init_block_pos_diff"]) > 1e-6)
    if mism:
        raise AssertionError(f"{chart_dir} vs {frozen_dir}: {mism} pairing mismatches")
    return [(bool(A[k].get("passthrough_success") or A[k].get("success")),
             bool(B[k].get("passthrough_success") or B[k].get("success")),
             A[k]["settled_block_pos_diff"], B[k]["settled_block_pos_diff"]) for k in ks]


def p_mcnemar(rows):
    """One-sided exact McNemar, H1: chart pass-through SR < frozen."""
    b = sum(1 for c, f, _, _ in rows if c and not f)      # chart-only wins
    c_ = sum(1 for c, f, _, _ in rows if f and not c)     # frozen-only wins
    n = b + c_
    return sum(comb(n, i) for i in range(b + 1)) / 2 ** n if n else 1.0


def p_wilcoxon(rows):
    """One-sided Wilcoxon signed-rank, H1: chart settled distance > frozen (chart worse)."""
    da = [r[2] for r in rows]
    db = [r[3] for r in rows]
    if all(x == y for x, y in zip(da, db)):
        return 1.0
    try:
        return float(wilcoxon(da, db, alternative="greater").pvalue)
    except ValueError:
        return 1.0


STATS = {"pass-through McNemar": p_mcnemar, "settled-dist Wilcoxon": p_wilcoxon}


def flag_rate(rows, stat, n, trials=TRIALS, seed=0):
    """Fraction of random n-episode probes that flag the adapter as harmful."""
    rnd = random.Random(seed)
    return sum(stat(rnd.sample(rows, n)) <= ALPHA for _ in range(trials)) / trials


def calibration(rows, stat, n, trials=TRIALS, seed=1):
    """False-positive rate under the exact paired null (random sign flip per pair).

    Flipping swaps the two arms for that task, which is the null a symmetric paired
    test assumes. A correctly calibrated one-sided test lands at ~ALPHA.
    """
    rnd = random.Random(seed)
    hits = 0
    for _ in range(trials):
        s = [(f, c, db, da) if rnd.random() < 0.5 else (c, f, da, db)
             for c, f, da, db in rnd.sample(rows, n)]
        hits += stat(s) <= ALPHA
    return hits / trials


def main():
    out = {"alpha": ALPHA, "trials": TRIALS, "ns": list(NS), "cases": {}}
    for label, (a, b) in CASES.items():
        rows = load_pairs(a, b)
        delta = st.mean(r[2] - r[3] for r in rows)
        entry = {"chart_dir": a, "frozen_dir": b, "n_pairs": len(rows),
                 "settled_delta_px": round(delta, 2),
                 "truth": "chart worse (SHOULD flag)" if delta > 0 else "chart better (must NOT flag)",
                 "pass_through_sr": [sum(r[0] for r in rows) / len(rows),
                                     sum(r[1] for r in rows) / len(rows)],
                 "screens": {}}
        for sname, stat in STATS.items():
            entry["screens"][sname] = {
                "flag_rate": {str(n): round(flag_rate(rows, stat, n), 4) for n in NS},
                "calibration_null": {str(n): round(calibration(rows, stat, n), 4) for n in NS},
            }
        out["cases"][label] = entry

    dst = os.path.join(ROOT, "day2_screen_power.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=1)

    hdr = "  ".join(f"n={n:<5d}" for n in NS)
    for label, e in out["cases"].items():
        print(f"\n{label}   [settled Δ {e['settled_delta_px']:+.1f} px — {e['truth']}]")
        print(f"    pass-through SR: chart {e['pass_through_sr'][0]:.2f} vs frozen {e['pass_through_sr'][1]:.2f}")
        print(f"    {'screen':24s} {hdr}")
        for sname, s in e["screens"].items():
            r = "  ".join(f"{100*s['flag_rate'][str(n)]:5.1f}%" for n in NS)
            c = "  ".join(f"{100*s['calibration_null'][str(n)]:5.1f}%" for n in NS)
            print(f"    {sname:24s} {r}")
            print(f"    {'  └ null (sign-flip)':24s} {c}")
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
