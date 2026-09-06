"""Widen the P0-G acceptance measurement from 40 test windows to every on-policy chunk.

The acceptance number (`eval_umf_chunkT2` 0.675 -> 0.558, -17%) rests on 8 test
trajectories / 40 T=2 windows, and best-val model selection rested on 8 more.
This recomputes the SAME quantity, paired per window, over train + val + test
(n = 580 windows), locally on the archived `trajs_R2.pt`. Read-only:
`atlas.score.umf`/`rollout_umf` and `atlas.stats` are called, never modified
(CLAUDE.md 1.2).

Paired by construction: frozen c0 and the chart score the identical window.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas.chart import Chart  # noqa: E402
from atlas.score import rollout_umf  # noqa: E402
from atlas.stats import paired_bootstrap  # noqa: E402

HUB = "hub/hub/facebookresearch_jepa-wms_main"


def windows(traj: dict, nas: int):
    enc, acts = traj["encoder_output"], traj["actions"]
    for i in range(0, acts.shape[0] - nas + 1):
        j = i + nas
        yield i, j, enc[i:j + 1], acts[i:j], traj["proprio"][i:i + 1].unsqueeze(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajs", type=Path, default=Path("phase0_v3/p0g_onpolicy/trajs_R2.pt"))
    ap.add_argument("--chart", type=Path, default=Path("phase0_v3/p0g_onpolicy/chart_ln_act_R2.pt"))
    ap.add_argument("--nas", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path("phase0_v3/c2_widened_offline_umf.json"))
    args = ap.parse_args()

    model, _ = torch.hub.load(HUB, "dino_wm_pusht", source="local",
                              force_reload=False, trust_repo=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    wm = model.model
    for m in wm.predictor.modules():
        if hasattr(m, "use_sdpa"):
            m.use_sdpa = True
    torch.set_float32_matmul_precision("high")

    blob = torch.load(args.trajs, map_location="cpu", weights_only=False)
    print("guard:", {k: blob["guard"][k] for k in sorted(blob["guard"])})

    def to_dev(t):
        return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in t.items()}

    splits = {s: [to_dev(t) for t in blob[s]] for s in ("train", "val", "test") if s in blob}
    for s, ts in splits.items():
        print(f"  split {s}: {len(ts)} trajectories")

    # Frozen c0 pass -- the predictor is pristine here (nothing applied yet).
    rows = []
    with torch.no_grad():
        for s, ts in splits.items():
            for ti, t in enumerate(ts):
                for i, j, enc, acts, pr in windows(t, args.nas):
                    rows.append({"split": s, "traj": ti, "w": [i, j],
                                 "umf_c0": rollout_umf(model, enc, acts, proprio_ctxt=pr),
                                 "latent_disp": (enc[-1] - enc[0]).norm(p="fro").item()})
    print(f"frozen c0 pass done: {len(rows)} windows")

    chart = Chart.load(str(args.chart), wm.predictor)
    chart.apply_(wm.predictor)
    k = 0
    with torch.no_grad():
        for s, ts in splits.items():
            for ti, t in enumerate(ts):
                for i, j, enc, acts, pr in windows(t, args.nas):
                    assert rows[k]["split"] == s and rows[k]["traj"] == ti and rows[k]["w"] == [i, j]
                    rows[k]["umf_chart"] = rollout_umf(model, enc, acts, proprio_ctxt=pr)
                    k += 1
    print(f"chart pass done: {k} windows (paired 1:1 with the frozen pass)")

    def report(sel, name):
        r = [x for x in rows if sel(x)
             and x["umf_c0"] is not None and x["umf_chart"] is not None]
        if not r:
            print(f"  {name}: no windows"); return None
        a = np.array([x["umf_chart"] for x in r]); b = np.array([x["umf_c0"] for x in r])
        d, ci = paired_bootstrap(a, b, n=10000, seed=0)
        pct = 100.0 * d / b.mean()
        print(f"  {name:<28} n={len(r):4d}  chart {a.mean():.4f}  frozen {b.mean():.4f}  "
              f"delta {d:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]  ({pct:+.1f}%)  "
              f"chart better in {int((a < b).sum())}/{len(r)}")
        return {"name": name, "n": len(r), "chart": float(a.mean()), "frozen": float(b.mean()),
                "delta": float(d), "ci95": [float(ci[0]), float(ci[1])],
                "pct": float(pct), "chart_better": int((a < b).sum())}

    print(f"\n--- paired per-window UMF at T={args.nas} (ungated; negative delta = chart better) ---")
    out = [report(lambda x: x["split"] == s, f"{s} split") for s in splits]
    out.append(report(lambda x: True, "ALL train+val+test"))
    out.append(report(lambda x: x["split"] in ("train", "val"), "train+val (tau population)"))

    args.out.write_text(json.dumps(
        {"nas": args.nas, "guard": {k: str(v) for k, v in blob["guard"].items()},
         "summary": [o for o in out if o], "rows": rows}, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
