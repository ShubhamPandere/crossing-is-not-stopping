"""FINAL_FIVE_DAY_PLAN R2.2 item B — n=100 headline, three disjoint task sets + merged.

Pre-registered (IMPLEMENTATION_PLAN_V3 §8.8): report task sets 0-19 / 20-49 / 50-99
separately and merged (paired n=100). Primary = paired settled block-distance (Wilcoxon +
bootstrap CI) + neither-pass-through clean subset (R2.0-a convention). If 50-99 does not
replicate the direction, §4 weakens to "two of three task sets" — do not average away.

nas=2 only (item B). Read-only.

CORRECTED 2026-09-03 (EVIDENCE_LEDGER.md N15-n100, the "🛑 RESOLVED CORRECTION" row, FIXLOG
V3-24): the original SETS below pointed at cells run before the `--fix-steps-left` planning-
horizon bug was found and fixed. The ledger explicitly states "This is now the CURRENT §4
headline value — use this row, not the archived one" and "the archived numbers above must not
be quoted as current." SETS now points at the `_fixsteps` cells so this script reproduces the
CURRENT, corrected headline (merged n=100 Δ≈-25.6px), not the retracted Δ-47.0px value.
Writes phase0_v3/damping_dose_n100_fixsteps.json (kept separate from the archived
day2_n100_analysis.json already on disk, which is retained for the historical record only).
"""
import json, random, statistics as st
from pathlib import Path
from scipy.stats import wilcoxon, binomtest

P0 = Path("phase0_v3")
random.seed(0)

SETS = {
    "0-19":  ("c2_settle2_baseline_nas2_fixsteps", "c2_settle2_ln_act_nas2_fixsteps"),
    "20-49": ("n50_baseline_nas2_ep20-49_fixsteps", "n50_ln_act_nas2_ep20-49_fixsteps"),
    "50-99": ("n100_baseline_nas2_ep50-99_fixsteps", "n100_ln_act_nas2_ep50-99_fixsteps"),
}


def rows(d):
    f = next((P0 / d).glob("*.jsonl"))
    return sorted((json.loads(l) for l in f.open()), key=lambda r: r["episode"])


def sdist(r):
    tr = r.get("settled_trace") or []
    return tr[-1]["block_pos_diff"] if tr else r.get("settled_block_pos_diff")


def boot_ci(x, n=20000):
    m = len(x)
    b = sorted(st.mean(random.choice(x) for _ in range(m)) for _ in range(n))
    return [round(b[int(0.025 * n)], 1), round(b[int(0.975 * n)], 1)]


def analyse(frozen, chart, label):
    mism = sum(1 for a, b in zip(frozen, chart)
               if abs(a["init_block_pos_diff"] - b["init_block_pos_diff"]) > 1e-6)
    fd = [sdist(r) for r in frozen]
    cd = [sdist(r) for r in chart]
    delta = [c - f for c, f in zip(cd, fd)]                 # chart - frozen; <0 = chart closer
    w = wilcoxon(cd, fd)
    # neither-pass-through subset (R2.0-a)
    ns = [i for i in range(len(chart)) if not chart[i]["success"] and not frozen[i]["success"]]
    nsd = [delta[i] for i in ns]
    w_ns = wilcoxon([cd[i] for i in ns], [fd[i] for i in ns]) if len(ns) > 1 else None
    # pass-through McNemar (one-sided, chart worse)
    b = sum(1 for c, f in zip(chart, frozen) if c["success"] and not f["success"])
    c_ = sum(1 for c, f in zip(chart, frozen) if f["success"] and not c["success"])
    mcn = binomtest(b, b + c_, 0.5, alternative="less").pvalue if (b + c_) else 1.0
    return {
        "label": label, "n": len(chart), "pairing_mismatches": mism,
        "settled_dist_chart_mean": round(st.mean(cd), 1),
        "settled_dist_frozen_mean": round(st.mean(fd), 1),
        "paired_delta_mean": round(st.mean(delta), 1),
        "paired_delta_ci95": boot_ci(delta),
        "chart_closer_count": f"{sum(1 for d in delta if d < 0)}/{len(delta)}",
        "wilcoxon_p": round(w.pvalue, 4),
        "direction": "chart closer" if st.mean(delta) < 0 else "chart farther",
        "clean_subset_neither_passthrough": {
            "n": len(ns), "delta_mean": round(st.mean(nsd), 1) if nsd else None,
            "wilcoxon_p": round(w_ns.pvalue, 4) if w_ns else None,
        },
        "passthrough_SR": [round(sum(r["success"] for r in frozen) / len(frozen), 3),
                           round(sum(r["success"] for r in chart) / len(chart), 3)],
        "passthrough_mcnemar_p_chart_worse": round(mcn, 4),
    }


out = {"per_task_set": {}, "merged_n100": None}
allf, allc = [], []
for name, (fdir, cdir) in SETS.items():
    fr, ch = rows(fdir), rows(cdir)
    allf += fr; allc += ch
    out["per_task_set"][name] = analyse(fr, ch, name)
out["merged_n100"] = analyse(allf, allc, "merged n=100")

d5099 = out["per_task_set"]["50-99"]["paired_delta_mean"]
reps = [out["per_task_set"][k]["paired_delta_mean"] < 0 for k in SETS]
out["verdict"] = (
    f"50-99 replicates (delta={d5099} < 0). All three task sets show chart closer. "
    "Merge and report paired n=100."
    if d5099 < 0 else
    f"50-99 does NOT replicate (delta={d5099} >= 0). Per plan §8.8: report the three sets "
    f"separately; §4 weakens to 'significant in {sum(reps)} of 3 task sets'. Do not average."
)

(P0 / "damping_dose_n100_fixsteps.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
