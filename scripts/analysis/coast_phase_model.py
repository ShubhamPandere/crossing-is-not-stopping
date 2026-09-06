"""FINAL_FIVE_DAY_PLAN R2.2 item C — analytical coast (residual-momentum) model.

pymunk applies `v *= damping^dt` each substep (pusht_env.py:436-437, 494; sim_hz=100,
control_hz=10 -> 1 env step = 0.1 s). A body coasting from speed v0 therefore travels

    D(T) = v0 * (1 - damping^T) / ln(1/damping)     [T seconds]
    D_inf = v0 / ln(1/damping)                       [asymptotic]

Two checks against `settled_trace` already on disk (read-only, no production path):

  1. per-episode: predict the step-1->40 coast from the step-1->5 early slope, compare.
  2. the ladder: predict settled-SR among pass-through crossings (block survives iff
     dist_at_crossing + remaining_coast <= 20 px), overlay on the measured curve.

Writes phase0_v3/day2_coast_model.json + phase0_v3/day2_fig_coast.{png,pdf}.
"""
import json, math, statistics as st
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P0 = Path("phase0_v3")
DT_ENV = 10 / 100          # 0.1 s per raw env step  (n_steps / sim_hz)
GOAL_R = 20.0

LADDER = {
    0.05: {2: "ladder_dmp005_baseline_nas2", 6: "ladder_dmp005_baseline_nas6"},
    0.1:  {2: "c2_settle2_dmp01_baseline_nas2", 6: "c2_settle2_dmp01_baseline_nas6"},
    0.2:  {2: "ladder_dmp02_baseline_nas2", 6: "ladder_dmp02_baseline_nas6"},
    0.3:  {2: "ladder_dmp03_baseline_nas2", 6: "ladder_dmp03_baseline_nas6"},
    0.5:  {2: "c2_settle2_baseline_nas2", 6: "c2_settle2_baseline_nas6"},
}   # damping 0 excluded: ln(1/1)=0, no coast; the R0 arms already show trace flat.


def rows(d):
    f = next((P0 / d).glob("*.jsonl"))
    return [json.loads(l) for l in f.open()]


def trace_map(r):
    return {c["step"]: c["block_pos_diff"] for c in (r.get("settled_trace") or [])}


def trace_angle(r):
    return {c["step"]: c["block_angle_diff"] for c in (r.get("settled_trace") or [])}


def v_at_step1(tr):
    """Signed speed (px/s) at hold-step 1, from the 1->5 slope, de-averaged.
    avg over the window = v1 * (1 - d^Tw)/(ln(1/d)*Tw); invert for v1. Sign: +ve = moving
    AWAY from goal (block_pos_diff increasing)."""
    if 1 not in tr or 5 not in tr:
        return None
    return (tr[5] - tr[1]) / (4 * DT_ENV)          # window-average speed (good enough; the
                                                   # de-averaging factor cancels in D_pred below)


def d_pred_from_slope(v_avg_1_5, damping):
    """Asymptotic coast from hold-step 1, predicted from the 1->5 average speed.
    D_inf = v1/ln(1/d); v1 = v_avg * ln(1/d)*Tw/(1-d^Tw)  =>  D_inf = v_avg*Tw/(1-d^Tw)."""
    Tw = 4 * DT_ENV
    return v_avg_1_5 * Tw / (1 - damping ** Tw)


out = {"dt_env_s": DT_ENV, "goal_radius_px": GOAL_R, "per_cell": {}, "ladder_survival": {},
       "disjoint_fit": {}}
scatter = {"pred": [], "meas": [], "damping": []}
disj = {"pred": [], "meas": []}   # PRIMARY: fit v0 on 5->15, predict the FINITE 15->40 coast


def _pearson(a, b):
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else None

for damp, cad in LADDER.items():
    for nas, d in cad.items():
        rs = rows(d)
        preds, meass, ratios = [], [], []
        ratios5 = []   # alternative: predict 5->40 coast from the 5->15 slope (clean free-coast)
        cross_pred_survive, cross_meas_survive, n_cross = 0, 0, 0
        for r in rs:
            tr = trace_map(r)
            if not {1, 5, 40} <= set(tr):
                continue
            v_avg = v_at_step1(tr)
            D_pred = d_pred_from_slope(v_avg, damp)
            D_meas = tr[40] - tr[1]
            preds.append(D_pred); meass.append(D_meas)
            if abs(D_meas) > 3:                       # skip near-static (ratio undefined)
                ratios.append(D_pred / D_meas)
            scatter["pred"].append(D_pred); scatter["meas"].append(D_meas); scatter["damping"].append(damp)
            if 15 in tr:
                v5 = (tr[15] - tr[5]) / (10 * DT_ENV)
                D_pred5 = v5 * (10 * DT_ENV) / (1 - damp ** (10 * DT_ENV))
                D_meas5 = tr[40] - tr[5]
                if abs(D_meas5) > 3:
                    ratios5.append(D_pred5 / D_meas5)
                # --- PRIMARY disjoint fit: v0 from 5->15, predict the FINITE 15->40 coast
                #     (fit window and predicted window share no data) ---
                Tw = 10 * DT_ENV
                v5_da = (tr[15] - tr[5]) * math.log(1 / damp) / (1 - damp ** Tw)  # de-avg to v(step5)
                v15 = v5_da * damp ** Tw                                          # decay to v(step15)
                Tp = 25 * DT_ENV                                                  # 15 -> 40
                d_pred_disj = v15 * (1 - damp ** Tp) / math.log(1 / damp)
                disj["pred"].append(d_pred_disj); disj["meas"].append(tr[40] - tr[15])
            # check 2: pass-through crossings only. Predicted-survive = translational coast
            # keeps it inside 20 px AND the block's rotation at hold-step 1 is already < 20 deg
            # (no rotational-coast model; a mis-rotated crossing can't settle-succeed regardless).
            if r.get("passthrough_success"):
                n_cross += 1
                ta = trace_angle(r)
                # predict from the CLEAN free-coast slope (5->15) where available -- at step 1
                # the block still has brief post-crossing contact (ratio 0.9 vs 1.00 at step 5)
                if 15 in tr:
                    v5 = (tr[15] - tr[5]) / (10 * DT_ENV)
                    Dc = v5 * (10 * DT_ENV) / (1 - damp ** (10 * DT_ENV))
                    final_pred = tr[5] + max(Dc, 0.0)
                else:
                    final_pred = tr[1] + max(D_pred, 0.0)
                ang_ok = ta.get(1, 9.9) < math.radians(20)
                cross_pred_survive += (final_pred <= GOAL_R and ang_ok)
                cross_meas_survive += bool(r.get("settled_success"))
        # pooled Pearson r on (pred, meas)
        r_pearson = None
        if len(preds) > 2:
            mp, mm = st.mean(preds), st.mean(meass)
            cov = sum((p - mp) * (m - mm) for p, m in zip(preds, meass))
            sp = math.sqrt(sum((p - mp) ** 2 for p in preds))
            sm = math.sqrt(sum((m - mm) ** 2 for m in meass))
            r_pearson = cov / (sp * sm) if sp and sm else None
        out["per_cell"][f"damp{damp}_nas{nas}"] = {
            "n": len(preds),
            "coast_pred_median": round(st.median(preds), 1) if preds else None,
            "coast_meas_median": round(st.median(meass), 1) if meass else None,
            "pred_over_meas_median": round(st.median(ratios), 2) if ratios else None,
            "pred_over_meas_median_from_step5": round(st.median(ratios5), 2) if ratios5 else None,
            "pearson_r_pred_meas": round(r_pearson, 3) if r_pearson is not None else None,
            "n_crossings": n_cross,
            "settled_SR_measured": round(cross_meas_survive / len(rs), 3),
            "settled_SR_predicted": round(cross_pred_survive / len(rs), 3),
        }

# ladder survival table (nas -> {damping -> (measured, predicted)})
for nas in (2, 6):
    out["ladder_survival"][f"nas{nas}"] = {
        str(damp): {
            "measured": out["per_cell"][f"damp{damp}_nas{nas}"]["settled_SR_measured"],
            "predicted": out["per_cell"][f"damp{damp}_nas{nas}"]["settled_SR_predicted"],
        } for damp in LADDER
    }

out["disjoint_fit"] = {
    "n": len(disj["pred"]),
    "pearson_r": round(_pearson(disj["pred"], disj["meas"]), 3),
    "pred_over_meas_median": round(st.median(p / m for p, m in zip(disj["pred"], disj["meas"])
                                             if abs(m) > 1), 2),
    "note": "PRIMARY: v0 fitted on hold-steps 5->15, predicting the finite 15->40 coast -- "
            "fit and predicted windows share NO data (the shipped per_cell/scatter numbers use "
            "a 1->5 fit that overlaps the 1->40 predicted quantity by ~24% at damping 0.5 and "
            "are kept only as the superseded overlapping version).",
}

(P0 / "day2_coast_model.json").write_text(json.dumps(out, indent=2))

# ---- figure ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
lim = max(max(map(abs, scatter["pred"])), max(map(abs, scatter["meas"]))) * 1.05
ax[0].plot([-lim, lim], [-lim, lim], "k--", lw=1, alpha=0.6, label="y = x")
sc = ax[0].scatter(scatter["meas"], scatter["pred"], c=scatter["damping"], cmap="viridis", s=22)
import statistics as _st
_r1 = _st.median(v["pred_over_meas_median"] for v in out["per_cell"].values())
_r5 = _st.median(v["pred_over_meas_median_from_step5"] for v in out["per_cell"].values())
_rp = _st.median(v["pearson_r_pred_meas"] for v in out["per_cell"].values())
ax[0].set_xlabel("measured coast, hold step 1→40 (px)")
ax[0].set_ylabel("predicted coast  v₀ / ln(1/damping)  (px)")
ax[0].set_title("C-1: closed-form coast vs measured, per episode")
ax[0].text(0.03, 0.97, f"median Pearson r = {_rp:.2f}\npred/meas = {_r1:.2f} (v₀ @ step 1)\n"
                       f"           = {_r5:.2f} (v₀ @ step 5, clean coast)",
           transform=ax[0].transAxes, va="top", fontsize=8,
           bbox=dict(boxstyle="round", fc="white", alpha=0.8))
ax[0].legend(fontsize=8, loc="lower right"); ax[0].grid(alpha=0.3)
fig.colorbar(sc, ax=ax[0], label="damping")

damps = list(LADDER)
for nas, style in [(6, "-o"), (2, "--s")]:
    m = [out["ladder_survival"][f"nas{nas}"][str(x)]["measured"] for x in damps]
    p = [out["ladder_survival"][f"nas{nas}"][str(x)]["predicted"] for x in damps]
    ax[1].plot(damps, m, style, color="tab:red", label=f"measured settled SR, nas={nas}")
    ax[1].plot(damps, p, style, color="tab:blue", label=f"predicted (coast model), nas={nas}")
ax[1].set_xlabel("R2 damping"); ax[1].set_ylabel("settled SR (n=20)")
ax[1].set_title("C-2: coast model predicts the ladder"); ax[1].set_ylim(-0.03, 1.03)
ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"phase0_v3/day2_fig_coast.{ext}", dpi=130)

print(json.dumps(out, indent=2))
print("\nwrote phase0_v3/day2_coast_model.json + day2_fig_coast.{png,pdf}")
