"""
scripts/phase0_measure.py — IMPLEMENTATION_PLAN_V3 §11.1 Phase 0, gates P0-A/B/D/E.

Forward-only (no CEM planner). Replays real Push-T demonstrations under R0/R1/R2
via load_regime_trajectories(source="dataset"), chunks each trajectory into
T = num_act_stepped model-steps, and for every chunk records:

  * umf_c0        — UMF of the frozen predictor c0 (score.rollout_umf, no chart)
  * e1_c0         — one-step error ‖ẑ1 − z1‖²           (router._e1_score, identity chart)
  * sdyn_c0       — negative cosine of Δz               (router._sdyn_score, identity chart)
  * latent_disp   — ‖z_T − z_0‖_F
  * block_disp_px — ‖block_xy(T) − block_xy(0)‖  (pixels, from info["block_pose"])

From those rows it derives:
  P0-B  motion gate      = P95 latent_disp over chunks with block_disp_px < 1 px
  P0-A  τ                = P95 umf_c0 over R0 chunks with latent_disp > motion_gate
  P0-D  strike rate      = frac(umf_c0 > τ) over informative chunks, per regime
        q                = smallest q with p_R0**q * N_STREAM_CHUNKS < 1
  P0-E  σ_r              = IQR of each router's score over the R0 informative set

Everything is written under --out (default phase0_v3/), never atlas_out/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import determinism as _determinism  # noqa: E402  — sets CUBLAS_WORKSPACE_CONFIG before torch

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "scripts" / "collection"))

import atlas
from atlas.chart import Chart
from atlas.score import rollout_umf
from atlas.router import _e1_score, _sdyn_score
from chart_training import load_regime_trajectories

# §11.2: ~1,100 scored chunks across a full E4 stream — used to derive q.
N_STREAM_CHUNKS = 1100


def load_model(device: str):
    _home = os.environ.get("ATLAS_HOME", str(atlas.ATLAS_HOME))
    hub_path = str(Path(_home) / "hub" / "hub" / "facebookresearch_jepa-wms_main")
    model, prep = torch.hub.load(hub_path, "dino_wm_pusht", source="local",
                                 force_reload=False, trust_repo=True)
    wrapper = model.to(device)
    wm = wrapper.model if hasattr(wrapper, "model") else wrapper
    for m in wm.predictor.modules():
        if hasattr(m, "use_sdpa"):
            m.use_sdpa = True
    torch.set_float32_matmul_precision("high")
    return wrapper, wm, prep


@torch.no_grad()
def score_chunks(wrapper, wm, prep, regime, num_trajs, traj_len, nas, device):
    c0 = Chart(wm.predictor, "ln_act")  # fresh clone == identity baseline
    trajs = load_regime_trajectories(
        wrapper, prep, regime, num_trajs=num_trajs, traj_len=traj_len,
        device=device, source="dataset", data_split="train",
        record_block_pose=True,
    )
    rows = []
    for ti, tr in enumerate(trajs):
        enc = tr["encoder_output"]           # [M+1, N, D]
        acts = tr["actions"]                 # [M, 10]
        prop = tr["proprio"]                 # [M+1, P_tok, D_p]
        bpose = tr["block_pose"]             # [M+1, 3]
        M = acts.shape[0]
        for k in range(0, M - nas + 1, nas):
            e = enc[k:k + nas + 1]
            a = acts[k:k + nas]
            pc = prop[k:k + 1].unsqueeze(0)  # [1, 1, P_tok, D_p]
            umf_v = rollout_umf(wrapper, e, a, proprio_ctxt=pc, motion_gate=None)
            e1_v = _e1_score(c0, wrapper, e, a, None, pc)
            sd_v = _sdyn_score(c0, wrapper, e, a, None, pc)
            rows.append({
                "regime": regime, "traj": ti, "k": k,
                "umf_c0": umf_v, "e1_c0": e1_v, "sdyn_c0": sd_v,
                "latent_disp": float((e[-1] - e[0]).norm(p="fro").item()),
                "block_disp_px": float(np.linalg.norm(bpose[k + nas][:2] - bpose[k][:2])),
                "seed": int(tr["seed"]), "episode_idx": tr.get("episode_idx"),
            })
    return rows


def iqr(x):
    x = np.asarray(x, dtype=float)
    return float(np.percentile(x, 75) - np.percentile(x, 25))


def derive(rows, out_dir):
    R = {r: [x for x in rows if x["regime"] == r] for r in ("R0", "R1", "R2")}

    # ── P0-B: motion gate ────────────────────────────────────────────────────
    static = [x["latent_disp"] for x in rows if x["block_disp_px"] < 1.0]
    motion_gate = float(np.percentile(static, 95)) if static else float("nan")

    # ── P0-A: τ ──────────────────────────────────────────────────────────────
    r0_inf = [x for x in R["R0"] if x["latent_disp"] > motion_gate and x["umf_c0"] is not None]
    tau = float(np.percentile([x["umf_c0"] for x in r0_inf], 95)) if r0_inf else float("nan")

    # ── P0-D: strike rate per regime + q ─────────────────────────────────────
    strike = {}
    for r, xs in R.items():
        inf = [x for x in xs if x["latent_disp"] > motion_gate and x["umf_c0"] is not None]
        strike[r] = {
            "n_informative": len(inf), "n_total": len(xs),
            "gated_frac": 1 - len(inf) / max(len(xs), 1),
            "strike_rate": (sum(x["umf_c0"] > tau for x in inf) / len(inf)) if inf else float("nan"),
        }
    p_r0 = strike["R0"]["strike_rate"]
    q = None
    if p_r0 and p_r0 > 0:
        q = 1
        while (p_r0 ** q) * N_STREAM_CHUNKS >= 1:
            q += 1

    # ── P0-E: σ_r (IQR) over R0 informative set ──────────────────────────────
    sigma = {
        "umf": iqr([x["umf_c0"] for x in r0_inf if x["umf_c0"] is not None]),
        "e1": iqr([x["e1_c0"] for x in r0_inf if x["e1_c0"] is not None]),
        "sdyn": iqr([x["sdyn_c0"] for x in r0_inf if x["sdyn_c0"] is not None]),
    }

    summary = {
        "config": {"n_stream_chunks_assumed": N_STREAM_CHUNKS},
        "P0-B_motion_gate": {"value": motion_gate, "n_static_chunks": len(static),
                             "note": "P95 latent ||z_T - z_0||_F over block_disp_px < 1px chunks"},
        "P0-A_tau": {"value": tau, "n_R0_informative": len(r0_inf),
                     "note": "P95 UMF(c0) over R0 informative chunks"},
        "P0-D_strike_rates": strike,
        "P0-D_q": {"value": q, "p_R0": p_r0,
                   "note": "smallest q with p_R0**q * N_stream < 1"},
        "P0-E_sigma_r": sigma,
    }
    (out_dir / "phase0_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "phase0_v3")
    ap.add_argument("--regimes", nargs="+", default=["R0", "R1", "R2"])
    ap.add_argument("--num-trajs", type=int, default=80)
    ap.add_argument("--traj-len", type=int, default=60, help="raw env steps (÷5 = model steps)")
    ap.add_argument("--num-act-stepped", type=int, default=2, help="T of each scored chunk")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    _determinism.make_deterministic(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[phase0] device={device}  (deterministic mode on)", flush=True)
    wrapper, wm, prep = load_model(device)

    all_rows = []
    for regime in args.regimes:
        print(f"[phase0] scoring {regime} ...", flush=True)
        rows = score_chunks(wrapper, wm, prep, regime, args.num_trajs,
                            args.traj_len, args.num_act_stepped, device)
        all_rows.extend(rows)
        print(f"[phase0] {regime}: {len(rows)} chunks", flush=True)

    with (args.out / "phase0_chunks.jsonl").open("w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")

    summary = derive(all_rows, args.out)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
