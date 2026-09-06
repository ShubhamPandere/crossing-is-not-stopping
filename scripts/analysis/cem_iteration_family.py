"""FINAL_FIVE_DAY_PLAN R2.2 item A — controller-family rank inversion.

12 controllers (planner configs), all R2 damping=0.5, nas=2 unless noted, settle-40, seeds 0-19:
  it in {1,3,10,30} x kind in {baseline,ln_act}   (it=10 = the c2_settle2 cells on disk)
  alpha=0            x kind in {baseline,ln_act}   (proprio term off in the CEM cost)
  it=10/nas=6        x kind in {baseline,ln_act}   (the cadence pair on disk)

Rank by (i) pass-through SR and (ii) mean settled block-distance; Kendall tau between the two
orderings with a permutation CI. Report per-pair inversions. Read-only.

CAVEAT (EVIDENCE_LEDGER.md N16-controller-family / N16-fixed-ladder): the `--fix-steps-left`
correction (2026-09-03) was applied to the frozen-baseline PRIMARY iteration ladder only
(IT_LAD below, now pointed at the `_fixsteps` cells — this is "the version the paper's §4
uses"). No `_fixsteps` reruns exist for the ln_act/alpha0/nas6 cells in the 12-controller
CTRL/SECONDARY Kendall-tau matrix below, so that matrix is left on its original (pre-fix)
data — per the ledger, this secondary analysis is not expected to reverse qualitatively, but
its exact settled-distance numbers should be treated as unverified against the same bug.
Writes phase0_v3/cem_iteration_family_fixsteps.json (PRIMARY table now corrected; SECONDARY
matrix unchanged/uncorrected, as above; kept separate from the archived
day2_controller_family.json already on disk, retained for the historical record only).
"""
import json, random, statistics as st
from pathlib import Path
from itertools import permutations
from scipy.stats import kendalltau

P0 = Path("phase0_v3")
random.seed(0)

# label -> dir (chart/frozen resolved by 'kind' in the label)
CTRL = {
    "it1  baseline":       "fam_it1_baseline_nas2",
    "it1  ln_act":         "fam_it1_ln_act_nas2",
    "it3  baseline":       "fam_it3_baseline_nas2",
    "it3  ln_act":         "fam_it3_ln_act_nas2",
    "it10 baseline":       "c2_settle2_baseline_nas2",
    "it10 ln_act":         "c2_settle2_ln_act_nas2",
    "it30 baseline":       "fam_it30_baseline_nas2",
    "it30 ln_act":         "fam_it30_ln_act_nas2",
    "a0   baseline":       "fam_alpha0_baseline_nas2",
    "a0   ln_act":         "fam_alpha0_ln_act_nas2",
    "nas6 baseline":       "c2_settle2_baseline_nas6",
    "nas6 ln_act":         "c2_settle2_ln_act_nas6",
}


def rows(d):
    f = next((P0 / d).glob("*.jsonl"))
    return [json.loads(l) for l in f.open()]


def sdist(r):
    tr = r.get("settled_trace") or []
    return tr[-1]["block_pos_diff"] if tr else r.get("settled_block_pos_diff")


ctrls = {}
missing = []
for lbl, d in CTRL.items():
    if not (P0 / d).exists():
        missing.append(lbl); continue
    rs = rows(d)
    ctrls[lbl] = {
        "n": len(rs),
        "pass_through_SR": round(sum(r["success"] for r in rs) / len(rs), 3),
        "settled_dist_mean": round(st.mean(sdist(r) for r in rs), 2),
        "settled_SR": round(sum(1 for r in rs
                                if (r.get("settled_trace") or [{}])[-1].get("success")) / len(rs), 3),
    }

labels = list(ctrls)
sr = {l: ctrls[l]["pass_through_SR"] for l in labels}
sd = {l: ctrls[l]["settled_dist_mean"] for l in labels}

# rank: higher pass-through SR = better; LOWER settled distance = better
rank_sr = sorted(labels, key=lambda l: -sr[l])
rank_sd = sorted(labels, key=lambda l: sd[l])
sr_pos = {l: i for i, l in enumerate(rank_sr)}
sd_pos = {l: i for i, l in enumerate(rank_sd)}

# Kendall tau-b (tie-corrected) between the pass-through-SR quality ordering and the
# settled-distance quality ordering. Higher SR = better; LOWER distance = better, so negate
# distance to put both on a "higher = better" scale. tau < 0  <=>  the two criteria rank
# controller quality in OPPOSED order. Computed on the raw value arrays (scipy handles ties
# properly) -- NOT on hand-broken rank positions.
def _qtau(group):
    return kendalltau([sr[l] for l in group], [-sd[l] for l in group])
tau, p_analytic = _qtau(labels)
frozen_grp = [l for l in labels if "ln_act" not in l]
chart_grp = [l for l in labels if "ln_act" in l]
tau_frozen, p_frozen = _qtau(frozen_grp)   # within-group: the genuine family signal, n=6, underpowered
tau_chart, p_chart = _qtau(chart_grp)

# --- the ITERATION LADDER (frozen c0 only, nas=2, the same 20 paired tasks) — N16's primary ---
# CORRECTED (N16-fixed-ladder): fixsteps cells, "the version the paper's §4 uses".
IT_LAD = {1: "fam_it1_baseline_nas2_fixsteps", 3: "fam_it3_baseline_nas2_fixsteps",
          10: "c2_settle2_baseline_nas2_fixsteps", 30: "fam_it30_baseline_nas2_fixsteps"}
ladder = {}
lad_rows = {}
for it, d in IT_LAD.items():
    rs = rows(d); lad_rows[it] = sorted(rs, key=lambda r: r["episode"])
    ladder[it] = {
        "pass_through_SR": round(sum(r["success"] for r in rs) / len(rs), 3),
        "settled_dist_mean": round(st.mean(sdist(r) for r in rs), 1),
        "median_progress_init_minus_settled": round(
            st.median(r["init_block_pos_diff"] - sdist(r) for r in rs), 1),
        "moved_toward_goal": f'{sum(1 for r in rs if sdist(r) < r["init_block_pos_diff"])}/{len(rs)}',
        "contacts_mean": round(st.mean(r["total_contacts"] for r in rs), 2),
    }
from scipy.stats import wilcoxon as _wil
d1 = [sdist(r) for r in lad_rows[1]]; d30 = [sdist(r) for r in lad_rows[30]]
lad_mismatch = sum(1 for a, b in zip(lad_rows[1], lad_rows[30])
                   if abs(a["init_block_pos_diff"] - b["init_block_pos_diff"]) > 1e-6)
ladder["it1_vs_it30_paired"] = {
    "settled_delta_mean": round(st.mean(a - b for a, b in zip(d1, d30)), 1),
    "it1_closer": f"{sum(1 for a, b in zip(d1, d30) if a < b)}/20",
    "wilcoxon_p_one_sided": round(_wil(d30, d1, alternative="greater").pvalue, 8),
    "pairing_mismatches": lad_mismatch,
}

# permutation CI on tau: shuffle the settled-distance -> controller assignment
perm_taus = []
xs = [sr[l] for l in labels]
base_ys = [-sd[l] for l in labels]
for _ in range(20000):
    ys = base_ys[:]
    random.shuffle(ys)
    t, _ = kendalltau(xs, ys)
    perm_taus.append(t)
perm_taus.sort()
p_perm = sum(1 for t in perm_taus if abs(t) >= abs(tau)) / len(perm_taus)

# pairwise inversions: pairs where SR says A>B but settled-dist says B>A
inversions = []
for a, b in permutations(labels, 2):
    if sr[a] > sr[b] and sd[a] > sd[b]:            # a better on SR, worse on distance
        inversions.append(f"{a}  >SR  {b}   but  {a} farther ({sd[a]} vs {sd[b]})")

chart_sd_vals = [sd[l] for l in chart_grp]
frozen_sd_vals = [sd[l] for l in frozen_grp]
# count crossings in the settled-distance ordering (positions where a frozen controller
# sits between two chart controllers or vice-versa)
by_sd = sorted(labels, key=lambda l: sd[l])
kinds = ["chart" if "ln_act" in l else "frozen" for l in by_sd]
crossings = sum(1 for i in range(1, len(kinds)) if kinds[i] != kinds[i - 1])

out = {
    "PRIMARY_iteration_ladder": ladder,
    "controllers": ctrls,
    "missing": missing,
    "rank_by_passthrough_SR_best_first": [(l, sr[l]) for l in rank_sr],
    "rank_by_settled_distance_best_first": [(l, sd[l]) for l in rank_sd],
    "two_group_structure": {
        "frozen_passthrough_SR_range": [min(sr[l] for l in frozen_grp), max(sr[l] for l in frozen_grp)],
        "chart_passthrough_SR_range": [min(sr[l] for l in chart_grp), max(sr[l] for l in chart_grp)],
        "disjoint_on_passthrough": max(sr[l] for l in chart_grp) < min(sr[l] for l in frozen_grp),
        "chart_settled_dist_range": [min(chart_sd_vals), max(chart_sd_vals)],
        "frozen_settled_dist_range": [min(frozen_sd_vals), max(frozen_sd_vals)],
        "settled_ordering_crossings": crossings - 1,
    },
    "kendall_tau_b_overall": round(tau, 3),
    "kendall_tau_caveat": "the 12 controllers form two disjoint 6-blocks on pass-through SR "
                          "(all frozen > all chart), so the overall tau largely re-expresses "
                          "N15 (chart worse on pass-through, better on settled). Secondary.",
    "kendall_within_frozen": {"tau": round(tau_frozen, 3), "p": round(p_frozen, 3), "n": 6},
    "kendall_within_chart": {"tau": round(tau_chart, 3), "p": round(p_chart, 3), "n": 6},
    "kendall_p_analytic": round(p_analytic, 4),
    "kendall_p_permutation": round(p_perm, 4),
    "tau_perm_null_ci95": [round(perm_taus[500], 3), round(perm_taus[-500], 3)],
    "n_pairwise_inversions": len(inversions),
    "n_pairs_total": len(labels) * (len(labels) - 1) // 2,
    "inversions": inversions,
}
(P0 / "cem_iteration_family_fixsteps.json").write_text(json.dumps(out, indent=2))

# ---- figure: pass-through SR vs settled distance, one point per controller ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7.2, 5))
for lbl in labels:
    is_chart = "ln_act" in lbl
    ax.scatter(sr[lbl], sd[lbl], s=70, marker="s" if is_chart else "o",
               c="tab:red" if is_chart else "tab:blue",
               label=None)
    ax.annotate(lbl.strip(), (sr[lbl], sd[lbl]), fontsize=7,
                xytext=(4, 3), textcoords="offset points")
ax.set_xlabel("pass-through SR  (higher = 'better')")
ax.set_ylabel("mean settled block-distance, px  (lower = better)")
ax.set_title(f"12 controllers: pass-through-SR and settled-distance quality orderings disagree\n"
             f"Kendall τ_b = {tau:.2f}  (perm p = {p_perm:.3f}, null [{perm_taus[500]:.2f}, {perm_taus[-500]:.2f}])  "
             f"— but the 12 split into two disjoint SR blocks; within-group τ {tau_frozen:.2f}/{tau_chart:.2f} (n=6, n.s.)")
ax.scatter([], [], marker="o", c="tab:blue", label="frozen c₀")
ax.scatter([], [], marker="s", c="tab:red", label="ln_act chart")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"phase0_v3/cem_iteration_family_fixsteps.{ext}", dpi=130)

print(json.dumps(out, indent=2))
print("\nwrote phase0_v3/cem_iteration_family_fixsteps.json + cem_iteration_family_fixsteps.{png,pdf}")
