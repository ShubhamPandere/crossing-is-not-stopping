"""FINAL_FIVE_DAY_PLAN Day 1 — 1.C dose ladder + 1.B damping-0.1 transfer.

Read-only. Recomputes from raw per-episode JSONL.

CORRECTED 2026-09-03 (EVIDENCE_LEDGER.md B3-dose-ladder-fixed: "This is the version the
paper's §3 uses; the archived B3-dose-ladder closed-loop rows are superseded"). The nas=2
(closed-loop) cells below now point at the `_fixsteps` reruns; nas=6 (open-loop) cells are
unchanged because that path is structurally immune to the `steps_left` bug the fix addresses.
1.B's nas=2 frozen baseline at d=0.1 reuses `ladder_dmp01_baseline_nas2_fixsteps` per
EVIDENCE_LEDGER N18-dmp02-transfer's explicit statement that this is the correct fixsteps
reuse (there is no separate `c2_settle2_dmp01_baseline_nas2_fixsteps` cell).
Writes phase0_v3/damping_dose_ladder_fixsteps.json (kept separate from the archived
day1_ladder_analysis.json already on disk, retained for the historical record only).

1.C  H1 dose-response: pass-through SR vs settled SR vs damping, frozen c0, nas in {2,6}.
     Also coast (residual momentum) per damping for H6.
1.B  chart(ln_act, trained@0.5) vs frozen c0 at damping=0.1, paired, settled block-distance.
"""
import json, statistics as st
from pathlib import Path
from scipy.stats import wilcoxon

P0 = Path("phase0_v3")

def rows(rel):
    p = P0 / rel
    return [json.loads(l) for l in p.open()] if p.exists() else None

def settled_dist(r):
    # settled_trace last checkpoint (hold 40) block_pos_diff; fallback settled_block_pos_diff
    tr = r.get("settled_trace") or []
    return tr[-1]["block_pos_diff"] if tr else r.get("settled_block_pos_diff")

def settled_succ(r):
    tr = r.get("settled_trace") or []
    return bool(tr[-1]["success"]) if tr else bool(r.get("settled_success"))

def coast(r):
    tr = {c["step"]: c["block_pos_diff"] for c in (r.get("settled_trace") or [])}
    if 1 in tr and 40 in tr:
        return tr[40] - tr[1]
    return None

# ---------- 1.C dose ladder ----------
LADDER = {  # damping -> {nas: jsonl rel path}
    0.0:  {2: "c2_settle2_R0_baseline_nas2_fixsteps/baseline_R0.jsonl", 6: "c2_settle2_R0_baseline_nas6/baseline_R0.jsonl"},  # R0 == damping 0
    0.05: {2: "ladder_dmp005_baseline_nas2_fixsteps/baseline_R2.jsonl", 6: "ladder_dmp005_baseline_nas6/baseline_R2.jsonl"},
    0.1:  {2: "ladder_dmp01_baseline_nas2_fixsteps/baseline_R2.jsonl", 6: "c2_settle2_dmp01_baseline_nas6/baseline_R2.jsonl"},
    0.2:  {2: "ladder_dmp02_baseline_nas2_fixsteps/baseline_R2.jsonl", 6: "ladder_dmp02_baseline_nas6/baseline_R2.jsonl"},
    0.3:  {2: "ladder_dmp03_baseline_nas2_fixsteps/baseline_R2.jsonl", 6: "ladder_dmp03_baseline_nas6/baseline_R2.jsonl"},
    0.5:  {2: "c2_settle2_baseline_nas2_fixsteps/baseline_R2.jsonl", 6: "c2_settle2_baseline_nas6/baseline_R2.jsonl"},
}

out = {"1C_dose_ladder": {}, "1B_dmp01_transfer": {}}

for damp, cfg in LADDER.items():
    for nas, rel in cfg.items():
        if rel is None:
            continue
        rs = rows(rel)
        if not rs:
            out["1C_dose_ladder"][f"damp{damp}_nas{nas}"] = "MISSING"
            continue
        n = len(rs)
        has_settle = any(r.get("settled_trace") for r in rs)
        rec = {
            "n": n,
            "damping": damp, "nas": nas,
            "pass_through_SR": round(sum(r["success"] for r in rs) / n, 3),
            "regime_config": rs[0].get("regime_config"),
        }
        if has_settle:
            sd = [settled_dist(r) for r in rs if settled_dist(r) is not None]
            cs = [coast(r) for r in rs if coast(r) is not None]
            rec.update({
                "settled_SR": round(sum(settled_succ(r) for r in rs) / n, 3),
                "settled_dist_mean": round(st.mean(sd), 1) if sd else None,
                "settled_dist_median": round(st.median(sd), 1) if sd else None,
                "coast_median_px": round(st.median(cs), 1) if cs else None,
                "contacts_mean": round(st.mean(r["total_contacts"] for r in rs), 2),
            })
        out["1C_dose_ladder"][f"damp{damp}_nas{nas}"] = rec

# ---------- 1.B chart vs frozen at damping 0.1 ----------
CHART_1B = {2: "dmp01_transfer_ln_act_nas2_fixsteps", 6: "dmp01_transfer_ln_act_nas6"}
FROZEN_1B = {2: "ladder_dmp01_baseline_nas2_fixsteps", 6: "c2_settle2_dmp01_baseline_nas6"}
for nas in (2, 6):
    ch = rows(f"{CHART_1B[nas]}/ln_act_R2.jsonl")
    fr = rows(f"{FROZEN_1B[nas]}/baseline_R2.jsonl")
    if not ch or not fr:
        out["1B_dmp01_transfer"][f"nas{nas}"] = "MISSING"
        continue
    ch = sorted(ch, key=lambda r: r["episode"]); fr = sorted(fr, key=lambda r: r["episode"])
    # pairing assert on init_block_pos_diff
    mism = sum(1 for a, b in zip(ch, fr) if abs(a["init_block_pos_diff"] - b["init_block_pos_diff"]) > 1e-6)
    cd = [settled_dist(r) for r in ch]; fd = [settled_dist(r) for r in fr]
    delta = [c - f for c, f in zip(cd, fd)]
    w = wilcoxon(cd, fd) if any(d != 0 for d in delta) else None
    # neither-succeeded subset = neither PASS-THROUGH succeeded (the success that break()s the
    # loop and creates the unequal-compute confound; matches V3-21 convention)
    ns_idx = [i for i in range(len(ch)) if not ch[i]["success"] and not fr[i]["success"]]
    ns_delta = [delta[i] for i in ns_idx]
    w_ns = wilcoxon([cd[i] for i in ns_idx], [fd[i] for i in ns_idx]) if len(ns_idx) > 1 and any(x != 0 for x in ns_delta) else None
    out["1B_dmp01_transfer"][f"nas{nas}"] = {
        "n": len(ch), "pairing_mismatches": mism,
        "chart_pass_through_SR": round(sum(r["success"] for r in ch) / len(ch), 3),
        "frozen_pass_through_SR": round(sum(r["success"] for r in fr) / len(fr), 3),
        "chart_settled_SR": round(sum(settled_succ(r) for r in ch) / len(ch), 3),
        "frozen_settled_SR": round(sum(settled_succ(r) for r in fr) / len(fr), 3),
        "chart_settled_dist_mean": round(st.mean(cd), 1),
        "frozen_settled_dist_mean": round(st.mean(fd), 1),
        "paired_delta_mean": round(st.mean(delta), 1),
        "chart_better_count": sum(1 for d in delta if d < 0),
        "wilcoxon_p": round(w.pvalue, 4) if w else None,
        "neither_succeeded_n": len(ns_idx),
        "ns_delta_mean": round(st.mean(ns_delta), 1) if ns_delta else None,
        "ns_wilcoxon_p": round(w_ns.pvalue, 4) if w_ns else None,
    }

(P0 / "damping_dose_ladder_fixsteps.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
