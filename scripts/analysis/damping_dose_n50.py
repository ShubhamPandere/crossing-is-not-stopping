"""FINAL_FIVE_DAY_PLAN Day 1.D — N=50 replication on disjoint tasks 20-49.

Pre-registered target (§8.7): settled block-distance at nas=2, chart < frozen (replicating
V3-21's nas=2 Delta -59.8 px). Decision: if direction replicates on 20-49 -> merge, report
paired n=50 (disclose two-launch structure). If not -> report both sets separately.

Read-only, from raw JSONL. Writes phase0_v3/day1_n50_analysis.json.
"""
import json, statistics as st, random
from pathlib import Path
from scipy.stats import wilcoxon

P0 = Path("phase0_v3")
random.seed(0)

def rows(rel):
    p = P0 / rel
    return sorted((json.loads(l) for l in p.open()), key=lambda r: r["episode"]) if p.exists() else None

def sdist(r):
    tr = r.get("settled_trace") or []
    return tr[-1]["block_pos_diff"] if tr else r.get("settled_block_pos_diff")

def ssucc(r):
    tr = r.get("settled_trace") or []
    return bool(tr[-1]["success"]) if tr else bool(r.get("settled_success"))

def boot_ci(deltas, n=20000):
    m = len(deltas)
    xs = sorted(st.mean(random.choice(deltas) for _ in range(m)) for _ in range(n))
    return round(xs[int(0.025 * n)], 1), round(xs[int(0.975 * n)], 1)

def paired(chart, frozen, label):
    mism = sum(1 for a, b in zip(chart, frozen)
               if abs(a["init_block_pos_diff"] - b["init_block_pos_diff"]) > 1e-6)
    cd = [sdist(r) for r in chart]; fd = [sdist(r) for r in frozen]
    delta = [c - f for c, f in zip(cd, fd)]
    w = wilcoxon(cd, fd)
    # "neither succeeded" = neither PASS-THROUGH succeeded — pass-through is what break()s the
    # episode loop, so it is the only success that creates the unequal-compute confound this
    # subset controls for (matches V3-21 / FABLE5_DAY1_RESULTS convention; settled-success
    # terminates nothing and conditioning on it controls for nothing).
    ns = [i for i in range(len(chart)) if not chart[i]["success"] and not frozen[i]["success"]]
    nsd = [delta[i] for i in ns]
    w_ns = wilcoxon([cd[i] for i in ns], [fd[i] for i in ns]) if len(ns) > 1 else None
    return {
        "label": label, "n": len(chart), "pairing_mismatches": mism,
        "chart_settled_dist_mean": round(st.mean(cd), 1),
        "frozen_settled_dist_mean": round(st.mean(fd), 1),
        "paired_delta_mean": round(st.mean(delta), 1),
        "paired_delta_ci95": boot_ci(delta),
        "chart_better_count": sum(1 for d in delta if d < 0),
        "wilcoxon_p": round(w.pvalue, 4),
        "chart_passthrough_SR": round(sum(r["success"] for r in chart) / len(chart), 3),
        "frozen_passthrough_SR": round(sum(r["success"] for r in frozen) / len(frozen), 3),
        "chart_settled_SR": round(sum(ssucc(r) for r in chart) / len(chart), 3),
        "frozen_settled_SR": round(sum(ssucc(r) for r in frozen) / len(frozen), 3),
        "neither_succeeded_n": len(ns),
        "ns_delta_mean": round(st.mean(nsd), 1) if nsd else None,
        "ns_wilcoxon_p": round(w_ns.pvalue, 4) if w_ns else None,
    }

out = {}
for nas in (2, 6):
    fr0 = rows(f"c2_settle2_baseline_nas{nas}/baseline_R2.jsonl")       # tasks 0-19
    ch0 = rows(f"c2_settle2_ln_act_nas{nas}/ln_act_R2.jsonl")
    fr1 = rows(f"n50_baseline_nas{nas}_ep20-49/baseline_R2.jsonl")      # tasks 20-49
    ch1 = rows(f"n50_ln_act_nas{nas}_ep20-49/ln_act_R2.jsonl")
    res = {
        "tasks_0_19":  paired(ch0, fr0, "0-19 (archived V3-21)"),
        "tasks_20_49": paired(ch1, fr1, "20-49 (new, 1.D)"),
        "merged_0_49": paired(ch0 + ch1, fr0 + fr1, "merged n=50"),
    }
    d0 = res["tasks_0_19"]["paired_delta_mean"]
    d1 = res["tasks_20_49"]["paired_delta_mean"]
    res["replication_verdict"] = (
        "REPLICATES (both delta<0, chart closer)" if d0 < 0 and d1 < 0
        else f"DOES NOT REPLICATE (0-19 delta={d0}, 20-49 delta={d1}) -> report separately, do not merge"
    )
    out[f"nas{nas}"] = res

(P0 / "day1_n50_analysis.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
