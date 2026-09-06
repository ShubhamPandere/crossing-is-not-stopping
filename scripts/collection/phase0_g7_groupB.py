"""
scripts/phase0_g7_groupB.py — P0-F / G7 CALIBRATION-STAGE diagnostic, "Group B".

Still NOT the final G7. Like Group A this informs the τ + motion-gate + m freeze
decision; it asserts nothing. The final G7 evaluates ONE human-frozen config.

What Group B adds over Group A: a real multi-chart library and a real cross-regime
A/B/A/B stream, driven through the ACTUAL production components in the production
order — SCORE (route/score_all) -> SELECT (route + hysteresis) -> EXPAND
(Expander.record + Expander.maybe_expand, real candidate fit) -> REFINE
(atlas.loop.atlas_refine, one AdaJEPA step, strictly AFTER scoring). The only
production step omitted is EXECUTE (the CEM plan) — it does not feed the router /
probe / K_max liveness this gate measures, and it is the only part that needs a
GPU-heavy planner. Everything here runs on the local card, free.

Library: {c0, chart_R1, chart_R2} from atlas_out/e2_charts/ (post-rollout-fix
charts). CAVEAT: these were trained by the OLD dataset-replay/contact-filtered
collector, not C.3's on-policy one — so this is a realistic-but-not-final library.
For a *liveness* check ("do the mechanisms fire on a plausible stream") that is
enough; re-run against P0-G's on-policy charts before treating any rate as final.

Stream: A,B,A,B,A,B with A=R0, B=R2 (IMPLEMENTATION_PLAN_V3 §8.4), each segment
`--n-ep` dataset-replay trajectories under that regime, chunked at T=2, in true
temporal order across segment boundaries.

VERIFIED against atlas/ source: route() is atlas.router.route (its own
incumbent-normalised hysteresis, FIX_SPEC A1); Expander is the real class;
atlas_refine is atlas.loop.atlas_refine with a persistent per-chart Adam (as its
docstring requires). motion-gate inequality: atlas/score.py:87-89.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
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
from atlas.library import Library
from atlas.router import route
from atlas.expand import Expander, ExpansionConfig
from atlas.loop import atlas_refine
from chart_training import load_regime_trajectories

REGIME_LABELS = {"R0": 0, "R1": 1, "R2": 2}


def load_library_from_e0(
    charts_dir: Path, kind: str, regimes: list[str], predictor
) -> tuple[Library, dict[int, int]]:
    """
    Build a Library directly from the chart-training script's on-disk layout
    (chart_{kind}_{regime}.pt files in charts_dir) — that layout doesn't match
    Library.save()'s format (no library_meta.pt, different filenames), so
    Library.load() cannot be used against it directly. c0 (index 0) is a
    fresh identity chart built from the live (frozen) predictor; each
    regime's fine-tuned chart is loaded and added in order.
    """
    c0 = Chart(predictor, kind)
    library = Library(c0, max_size=max(10, len(regimes) + 1))
    label_to_chart: dict[int, int] = {}
    for regime in regimes:
        chart_path = charts_dir / f"chart_{kind}_{regime}.pt"
        if not chart_path.exists():
            raise FileNotFoundError(
                f"Chart not found: {chart_path}. Run scripts/collection/chart_training.py first, "
                f"or point --charts at a directory containing chart_{{kind}}_{{regime}}.pt files."
            )
        chart = Chart.load(chart_path, predictor)
        idx = library.add(chart)
        label_to_chart[REGIME_LABELS[regime]] = idx
    return library, label_to_chart


SEGMENTS = ["R0", "R2", "R0", "R2", "R0", "R2"]


def load_model(device):
    import os
    home = os.environ.get("ATLAS_HOME", str(atlas.ATLAS_HOME))
    hub = str(Path(home) / "hub" / "hub" / "facebookresearch_jepa-wms_main")
    model, prep = torch.hub.load(hub, "dino_wm_pusht", source="local",
                                 force_reload=False, trust_repo=True)
    wrapper = model.to(device)
    wm = wrapper.model if hasattr(wrapper, "model") else wrapper
    for m in wm.predictor.modules():
        if hasattr(m, "use_sdpa"):
            m.use_sdpa = True
    return wrapper, wm, prep


def build_stream(wrapper, prep, n_ep, traj_len, nas, device, seed_base=50_000):
    """One list of (regime, seg_idx, chunk_dict) in true temporal order."""
    stream = []
    for si, regime in enumerate(SEGMENTS):
        trajs = load_regime_trajectories(
            wrapper, prep, regime, num_trajs=n_ep, traj_len=traj_len,
            device=device, source="dataset", data_split="train",
            seed_offset=seed_base + si * 1000,  # disjoint per segment
        )
        for ti, tr in enumerate(trajs):
            enc, acts, prop = tr["encoder_output"], tr["actions"], tr["proprio"]
            M = acts.shape[0]
            tid = (si, ti)  # global trajectory id — a probe should verify on a
                            # chunk from the SAME trajectory (a continuous rollout),
                            # not the first chunk of the next episode/regime
            for k in range(0, M - nas + 1, nas):
                stream.append((regime, si, tid, {
                    "enc": enc[k:k + nas + 1],
                    "acts": acts[k:k + nas],
                    "pc": prop[k:k + 1].unsqueeze(0),
                }))
    return stream


@torch.no_grad()
def _score_umf(chart, wm_wrapper, chunk, motion_gate):
    from atlas.score import umf
    return umf(chart, wm_wrapper, chunk["enc"], chunk["acts"],
               motion_gate=motion_gate, proprio_ctxt=chunk["pc"])


def _chart_checksum(chart) -> str:
    h = hashlib.sha1()
    for n in sorted(chart._params):
        h.update(n.encode())
        h.update(chart._params[n].detach().cpu().numpy().tobytes())
    return h.hexdigest()[:12]


def run_stream(wrapper, wm, charts_dir, kind, stream, *, tau, motion_gate, m, k_max,
               cold_start=False, pristine=None, no_refine=False, no_expand=False,
               n_verify=1, n_probe=20):
    # Contamination guard: atlas_refine leaves the predictor holding the
    # last-refined chart's weights (loop.py:294-302), so a later config's
    # `Chart(wm.predictor, kind)` would snapshot a CONTAMINATED c0. Reload the
    # pristine checkpoint state before every config so all 8 rows start from
    # the identical baseline (same rule as run_e0.py's pristine_predictor_state).
    if pristine is not None:
        wm.predictor.load_state_dict(pristine)

    if cold_start:
        # {c0} only — the REAL S2 starting condition. Commits must happen here
        # for the continual-learning story to hold.
        c0 = Chart(wm.predictor, kind)
        library = Library(c0, max_size=k_max)
    else:
        library, _ = load_library_from_e0(charts_dir, kind, ["R1", "R2"], wm.predictor)
        library.max_size = k_max
        c0 = library[0]
    c0_sum = _chart_checksum(c0)
    c0_frozen = c0.clone()  # never refined — the honest baseline for the usefulness metric
    expander = Expander(ExpansionConfig(tau=tau, q=3, kind=kind, n_probe=n_probe))
    optims: dict[int, object] = {}

    cur = 0
    n_gated = n_infor = n_switch = n_switch_boundary = 0
    armed = committed = rej_score = rej_full = 0
    verify_same_traj = verify_cross_traj = 0
    sel_expanded = 0  # times the router picked a chart that didn't exist at stream start
    n_charts_at_start = len(library)
    max_lib = len(library)
    prev_seg = None
    sel_hist = []
    # USEFULNESS: per informative chunk, UMF of the router's selected chart vs
    # UMF of frozen c0 on the SAME chunk. If expansion+routing helps, sel < c0,
    # especially on R2 (where the committed charts should specialise). The
    # committed charts are pointless if this gap is ~0.
    sel_umf = {"R0": [], "R2": []}
    c0_umf = {"R0": [], "R2": []}
    # PER-COMMIT GENERALISATION: when a chart commits, snapshot it AND the
    # incumbent it beat, then check whether it keeps beating that incumbent on
    # the NEXT chunks of the same regime — or whether the verification chunk was
    # a one-off fluke (the "committed != helpful" risk).
    commit_events = []  # {idx, regime, cand: Chart, incumbent: Chart}

    for i, (regime, seg, tid, chunk) in enumerate(stream):
        # ── SCORE + SELECT (real router; its own hysteresis) ──────────────────
        sel, info = route("umf", library, wrapper, chunk["enc"], chunk["acts"],
                          current_idx=cur, motion_gate=motion_gate,
                          hysteresis=m, proprio_ctxt=chunk["pc"])
        gated = info.get("gated", False)
        scores = [s for s in info["scores"] if s is not None]
        best_umf = min(scores) if scores else None
        if gated or best_umf is None:
            n_gated += 1
        else:
            n_infor += 1
            if sel != cur:
                n_switch += 1
                if prev_seg is not None and seg != prev_seg:
                    n_switch_boundary += 1
        cur = sel
        sel_hist.append(sel)
        prev_seg = seg
        if sel >= n_charts_at_start:
            sel_expanded += 1

        # ── USEFULNESS: selected-chart UMF vs frozen-c0 UMF on this same chunk ──
        if not gated and best_umf is not None and regime in sel_umf:
            sel_score = info["scores"][sel]
            if sel_score is not None:
                c0_score = _score_umf(c0_frozen, wrapper, chunk, motion_gate)
                if c0_score is not None:
                    sel_umf[regime].append(sel_score)
                    c0_umf[regime].append(c0_score)

        # ── EXPAND (real Expander; real candidate fit + verify on the NEXT chunk).
        # Verify against the next chunk of the SAME trajectory where one exists
        # (a continuous rollout, as in the real S2 loop); fall back to the global
        # next only at a trajectory's end, and count how often that fallback
        # happens — a cross-trajectory verify chunk has a different init state and
        # makes 'beats_best on the next chunk' unfairly hard.
        expander.record(best_umf, chunk["enc"], chunk["acts"], chunk["pc"])
        nxt = None
        if i + 1 < len(stream) and stream[i + 1][2] == tid:
            nxt, cross = stream[i + 1][3], False
        elif i + 1 < len(stream):
            nxt, cross = stream[i + 1][3], True
        if nxt is not None and not no_expand:
            vc, vc_idx = [], []
            for j in range(i + 1, len(stream)):
                if len(vc) >= n_verify:
                    break
                if stream[j][0] == regime:  # next SAME-REGIME chunks
                    ck = stream[j][3]
                    vc.append((ck["enc"], ck["acts"], ck["pc"]))
                    vc_idx.append(j)
            # single-chunk default also consumes stream[i+1] as its verify chunk
            if n_verify <= 1 and i + 1 < len(stream):
                vc_idx = [i + 1]
            outcome = expander.maybe_expand(library, wrapper, nxt["enc"],
                                            nxt["acts"], motion_gate, nxt["pc"],
                                            verify_chunks=vc if (n_verify > 1 and vc) else None)
            if outcome != "not_ready":
                armed += 1
                verify_cross_traj += cross
                verify_same_traj += (not cross)
                if outcome == "committed":
                    committed += 1
                    inc_idx = expander._last_probe_debug.get("incumbent_idx", 0)
                    commit_events.append({
                        "idx": i, "regime": regime,
                        "cand": library[-1].clone(),          # snapshot at commit time
                        "incumbent": library[inc_idx].clone(),
                        "verify_idx": set(vc_idx),  # exclude these from the followup window
                    })
                elif outcome == "rejected_score":
                    rej_score += 1
                elif outcome == "rejected_full":
                    rej_full += 1
            max_lib = max(max_lib, len(library))

        # ── REFINE (real atlas_refine, AFTER scoring; persistent per-chart Adam)
        if not gated and not no_refine:
            if sel not in optims:
                import torch.optim as optim
                ch = library[sel]
                ch.apply_(wm.predictor)  # so lora params exist as named_parameters
                if ch.kind == "lora4":
                    ps = [p for n, p in wm.predictor.named_parameters()
                          if "lora_A" in n or "lora_B" in n]
                else:
                    ps = [p for n, p in wm.predictor.named_parameters() if n in ch._param_names]
                for p in ps:
                    p.requires_grad_(True)
                ch.restore_(wm.predictor)
                optims[sel] = optim.Adam(ps, lr=5e-4)
            atlas_refine(library[sel], wrapper, chunk["enc"], chunk["acts"],
                         lr=5e-4, proprio_ctxt=chunk["pc"], optimizer=optims[sel])

    # ── PER-COMMIT GENERALISATION post-pass (forward-only, no planner) ────────
    # For each commit: on the next FOLLOWUP same-regime chunks that were NOT part
    # of the verify set used to ACCEPT it, does the candidate keep beating the
    # incumbent? Excluding the verify chunks is essential — otherwise the check
    # is circular (accept because it won those chunks, "confirm" on the same
    # chunks). FOLLOWUP counts the disjoint chunks, so the window auto-extends
    # past however many verify chunks were consumed.
    FOLLOWUP = 10
    per_commit = []
    for ev in commit_events:
        # classify the SAME commit two ways from one run — the only clean
        # window-fix comparison. (The candidate/refine BACKWARD passes are
        # CUDA-nondeterministic — cudnn.deterministic=False — so cross-run
        # comparisons are confounded; within-run dual classification is not.
        # The predictor loads in eval() mode, so this is NOT a dropout issue.)
        #   disjoint  = skip the chunks used to ACCEPT this commit (correct)
        #   overlap   = the old buggy window that re-scored the accept chunks
        res = {}
        for mode, skip in (("disj", True), ("overlap", False)):
            wins = total = 0
            gaps = []
            for j in range(ev["idx"] + 1, len(stream)):
                if total >= FOLLOWUP:
                    break
                if skip and j in ev.get("verify_idx", ()):
                    continue
                reg_j, _, _, ck = stream[j]
                if reg_j != ev["regime"]:
                    continue
                u_cand = _score_umf(ev["cand"], wrapper, ck, motion_gate)
                u_inc = _score_umf(ev["incumbent"], wrapper, ck, motion_gate)
                if u_cand is None or u_inc is None:
                    continue
                total += 1
                wins += (u_cand < u_inc)
                gaps.append(u_inc - u_cand)
            res[mode] = {"n": total, "wf": (wins / total) if total else float("nan"),
                         "gap": float(np.mean(gaps)) if gaps else float("nan")}
        per_commit.append({
            "regime": ev["regime"],
            "n_followup": res["disj"]["n"], "win_frac": res["disj"]["wf"],
            "mean_gap": res["disj"]["gap"],
            "n_overlap": res["overlap"]["n"], "win_frac_overlap": res["overlap"]["wf"],
        })
    generalising = sum(1 for p in per_commit if p["n_followup"] >= 3 and p["win_frac"] > 0.5)
    onehit = sum(1 for p in per_commit if p["n_followup"] >= 3 and p["win_frac"] <= 0.5)

    N = n_gated + n_infor
    return {
        "tau": tau, "motion_gate_pctl": motion_gate, "m": m, "k_max": k_max,
        "per_commit": per_commit, "commits_generalising": generalising,
        "commits_onehit": onehit,
        "n_chunks": N, "n_gated": n_gated, "gate_fire_rate": n_gated / N,
        "n_informative": n_infor,
        "switch_rate": (n_switch / n_infor) if n_infor else float("nan"),
        "switches": n_switch, "switches_at_boundary": n_switch_boundary,
        "distinct_charts_selected": len(set(sel_hist)),
        "probe_armed": armed, "probe_committed": committed,
        "probe_rej_score": rej_score, "probe_rej_full": rej_full,
        "verify_same_traj": verify_same_traj, "verify_cross_traj": verify_cross_traj,
        "sel_expanded_chart": sel_expanded, "n_charts_at_start": n_charts_at_start,
        "max_library_size": max_lib, "k_max": k_max, "c0_checksum": c0_sum,
        # usefulness: mean UMF of the router's selected chart vs frozen c0, per regime
        "sel_umf_R2": float(np.mean(sel_umf["R2"])) if sel_umf["R2"] else float("nan"),
        "c0_umf_R2": float(np.mean(c0_umf["R2"])) if c0_umf["R2"] else float("nan"),
        "sel_umf_R0": float(np.mean(sel_umf["R0"])) if sel_umf["R0"] else float("nan"),
        "c0_umf_R0": float(np.mean(c0_umf["R0"])) if c0_umf["R0"] else float("nan"),
        "n_umf_R2": len(sel_umf["R2"]), "n_umf_R0": len(sel_umf["R0"]),
    }


def band(x, lo=0.05, hi=0.95):
    if x != x:  # nan
        return "n/a"
    return "OK" if lo <= x <= hi else ("DEAD ~0%" if x < lo else "SATURATED ~100%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--charts-dir", default=str(REPO / "atlas_out" / "e2_charts"))
    ap.add_argument("--kind", default="ln_act")
    ap.add_argument("--n-ep", type=int, default=6, help="trajectories per A/B segment")
    ap.add_argument("--traj-len", type=int, default=60)
    ap.add_argument("--nas", type=int, default=2)
    ap.add_argument("--taus", type=float, nargs="+", default=[0.262, 0.5])
    ap.add_argument("--gate-pctls", type=int, nargs="+", default=[50, 75])
    ap.add_argument("--ms", type=float, nargs="+", default=[0.0, 0.05])
    ap.add_argument("--k-max", type=int, default=5)
    ap.add_argument("--cold-start", action="store_true",
                    help="Library starts as {c0} only (the real S2 condition) — "
                         "tests whether the probe actually COMMITS new charts.")
    ap.add_argument("--no-refine", action="store_true",
                    help="Ablation: disable the REFINE step. If m=0.05 still kills "
                         "commits with REFINE off, the cause is the margin threshold "
                         "alone, not the incumbent-monopolises-refinement loop.")
    ap.add_argument("--n-probe", type=int, nargs="+", default=[20],
                    help="Candidate fit steps (ExpansionConfig.n_probe). Sweep to test "
                         "whether one-hit-wonder commits are a weak-fit problem vs a "
                         "weak-verification problem.")
    ap.add_argument("--n-verify", type=int, nargs="+", default=[1],
                    help="Accept-gate held-out chunks: 1 = current single-chunk rule; "
                         ">1 = candidate must beat incumbent on a MAJORITY of that many "
                         "next same-regime chunks (less noise-prone). Multiple values "
                         "=> sweep, to show the one-hit-wonder rate vs verification size.")
    ap.add_argument("--seed-base", type=int, default=50_000)
    ap.add_argument("--seed-bases", type=int, nargs="+", default=None,
                    help="Multiple stream draws — reports the DISTRIBUTION of commit "
                         "counts across independent episode draws (the C2-reliability "
                         "question). Overrides --seed-base.")
    ap.add_argument("--arms", nargs="+", default=["full"],
                    choices=["frozen", "refine", "full"],
                    help="frozen={c0} no refine/expand; refine={c0}+refine; "
                         "full={c0}+refine+expansion. Multiple => USEFULNESS comparison: "
                         "does expansion lower the selected chart's UMF on R2 vs frozen c0?")
    ap.add_argument("--out", default=str(REPO / "phase0_v3" / "g7_groupB_calibration.txt"))
    args = ap.parse_args()
    _ARM_FLAGS = {"frozen": dict(no_refine=True, no_expand=True),
                  "refine": dict(no_refine=False, no_expand=True),
                  "full": dict(no_refine=False, no_expand=False)}

    _determinism.make_deterministic(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = args.seed_bases if args.seed_bases else [args.seed_base]
    print(f"[g7-B] device={device}  cold_start={args.cold_start} no_refine={args.no_refine} "
          f"seeds={seeds}  (deterministic mode on)", flush=True)
    wrapper, wm, prep = load_model(device)
    pristine = copy.deepcopy(wm.predictor.state_dict())  # contamination guard

    chunks_rows = [json.loads(l) for l in (REPO / "phase0_v3" / "phase0_chunks.jsonl").open()]
    static = [r["latent_disp"] for r in chunks_rows if r["block_disp_px"] < 1.0]
    gate_vals = {p: float(np.percentile(static, p)) for p in args.gate_pctls}

    rows = []
    for sb in seeds:
        torch.manual_seed(sb)  # reproducible candidate-fit / Adam RNG per draw
        stream = build_stream(wrapper, prep, args.n_ep, args.traj_len, args.nas, device,
                              seed_base=sb)
        for gp in args.gate_pctls:
            for tau in args.taus:
                for m in args.ms:
                  for nv in args.n_verify:
                   for npr in args.n_probe:
                    for arm in args.arms:
                        r = run_stream(wrapper, wm, Path(args.charts_dir), args.kind, stream,
                                       tau=tau, motion_gate=gate_vals[gp], m=m, k_max=args.k_max,
                                       cold_start=args.cold_start, pristine=pristine,
                                       n_verify=nv, n_probe=npr, **_ARM_FLAGS[arm])
                        r.update(gate_pctl=gp, seed_base=sb, arm=arm, n_verify=nv, n_probe=npr,
                                 n_stream_chunks=len(stream))
                        rows.append(r)
                        print(f"  seed={sb} P{gp} m={m:g} nv={nv} np={npr} arm={arm:>6} "
                              f"[c0={r['c0_checksum']}] ({len(stream)}c): "
                              f"switch {100*r['switch_rate']:.0f}% "
                              f"({r['sel_expanded_chart']}x expanded) | "
                              f"armed {r['probe_armed']} commit {r['probe_committed']} "
                              f"(rej {r['probe_rej_score']}s/{r['probe_rej_full']}f) | "
                              f"maxlib {r['max_library_size']}/{r['k_max']} | "
                              f"UMF-R2 sel {r['sel_umf_R2']:.3f} vs c0 {r['c0_umf_R2']:.3f}",
                              flush=True)

    full_rows = [r for r in rows if r["arm"] == "full"]
    if args.seed_bases and full_rows:
        commits = [r["probe_committed"] for r in full_rows]
        armed = [r["probe_armed"] for r in full_rows]
        print(f"\n=== commit-count distribution over {len(commits)} full-arm draws ===")
        print(f"  commits: {sorted(commits)}   mean {np.mean(commits):.1f}  "
              f"zero-commit: {commits.count(0)}/{len(commits)}   armed: {sorted(armed)}")

    # ── DID THE COMMITS ACTUALLY HELP? per (n_verify, n_probe) ──────────────
    # Reports BOTH follow-up windows on the SAME commit set — the only clean
    # window-fix comparison (candidate/refine BACKWARD is CUDA-nondeterministic, so
    # cross-run comparisons are confounded).
    for nv in args.n_verify:
      for npr in args.n_probe:
        fr = [r for r in full_rows if r.get("n_verify", 1) == nv and r.get("n_probe", 20) == npr]
        pc_d = [p for r in fr for p in r["per_commit"] if p["n_followup"] >= 3]
        pc_o = [p for r in fr for p in r["per_commit"] if p.get("n_overlap", 0) >= 3]
        nc = sum(r["probe_committed"] for r in fr)
        gd = sum(1 for p in pc_d if p["win_frac"] > 0.5)
        go = sum(1 for p in pc_o if p["win_frac_overlap"] > 0.5)
        d = f"{gd}/{len(pc_d)} ({100*gd/len(pc_d):.0f}%)" if pc_d else "n/a"
        o = f"{go}/{len(pc_o)} ({100*go/len(pc_o):.0f}%)" if pc_o else "n/a"
        print(f"=== nv={nv} n_probe={npr}: {nc} commits | generalise DISJOINT {d}  "
              f"vs OVERLAP(buggy) {o}")
    if full_rows:
        allpc = [p for r in full_rows for p in r["per_commit"] if p["n_followup"] >= 3]
        gen = sum(1 for p in allpc if p["win_frac"] > 0.5)
        one = len(allpc) - gen
        print(f"\n=== PER-COMMIT GENERALISATION (next {10} same-regime chunks after each commit) ===")
        print(f"  {len(allpc)} commits with >=3 followup chunks:  "
              f"{gen} keep beating the incumbent (>50% of followups),  "
              f"{one} are one-hit wonders (verification chunk was a fluke)")
        if allpc:
            wf = [p['win_frac'] for p in allpc]
            mg = [p['mean_gap'] for p in allpc if p['mean_gap'] == p['mean_gap']]
            print(f"  win-fraction distribution: {[round(w,2) for w in sorted(wf)]}")
            print(f"  mean UMF gap (incumbent - candidate) over followups: "
                  f"{np.mean(mg):+.3f}  (positive => candidate really is better)")
        print("  If most commits are one-hit wonders, the accept criterion is noise-prone —")
        print("  a serious finding to have BEFORE E3/E4 spends budget on this mechanism.")

    if len(args.arms) > 1:
        print(f"\n=== USEFULNESS: mean UMF of the SELECTED chart, per arm "
              f"(lower = better prediction; the point of committing charts) ===")
        arm_r2 = {}
        for arm in args.arms:
            ar = [r for r in rows if r["arm"] == arm]
            for reg in ("R0", "R2"):
                s = np.nanmean([r[f"sel_umf_{reg}"] for r in ar])
                c = np.nanmean([r[f"c0_umf_{reg}"] for r in ar])  # frozen c0, same every arm
                print(f"  {arm:>6}  {reg}: selected {s:.3f}  vs FROZEN-c0 {c:.3f}  "
                      f"(gap {c - s:+.3f}, {100*(c-s)/c:+.0f}%)")
                if reg == "R2":
                    arm_r2[arm] = s
        if {"refine", "full"} <= arm_r2.keys():
            d = arm_r2["refine"] - arm_r2["full"]
            verdict = ("EXPANSION HELPS beyond refinement" if d > 0.005
                       else "expansion adds ~nothing beyond refinement (churn)")
            print(f"  full vs refine on R2: {arm_r2['full']:.3f} vs {arm_r2['refine']:.3f} "
                  f"-> {verdict}")

    lib_desc = "{c0} only (COLD START — real S2 condition)" if args.cold_start \
        else f"{{c0,R1,R2}} from {args.charts_dir}"
    csums = sorted({r["c0_checksum"] for r in rows})
    contam = "CLEAN (all configs identical c0)" if len(csums) == 1 \
        else f"CONTAMINATED — {len(csums)} distinct c0 checksums: {csums}"
    lines = ["G7 Group B — calibration-stage diagnostic (freezes nothing, asserts nothing)",
             f"{SEGMENTS}, library {lib_desc}, seeds {seeds}",
             f"no_refine={args.no_refine}  c0-baseline: {contam}",
             "CAVEAT: charts here are NOT P0-G on-policy. Confirms tau=0.5 dead / "
             "tau~0.26 alive; NOT the frozen decision.",
             ""]
    if args.seed_bases:
        commits = [r["probe_committed"] for r in rows]
        lines.append(f"COMMIT DISTRIBUTION over {len(rows)} draws: {sorted(commits)}  "
                     f"(mean {np.mean(commits):.1f}, zero-commit {commits.count(0)}/{len(commits)}, "
                     f"armed always {min(r['probe_armed'] for r in rows)}-"
                     f"{max(r['probe_armed'] for r in rows)})")
        lines.append("")
    for r in rows:
        lines.append(
            f"seed{r['seed_base']} P{r['gate_pctl']:>2}  tau={r['tau']:<5g} m={r['m']:<4g}  "
            f"motion-gate {band(r['gate_fire_rate']):>16} ({100*r['gate_fire_rate']:.0f}%)  "
            f"router-switch {band(r['switch_rate']):>16} ({100*r['switch_rate']:.0f}%, "
            f"{r['distinct_charts_selected']}/{r['n_charts_at_start']} start + "
            f"{r['sel_expanded_chart']}x expanded)  "
            f"probe: armed {r['probe_armed']} commit {r['probe_committed']} "
            f"(rej: {r['probe_rej_score']}score/{r['probe_rej_full']}full; "
            f"verify {r['verify_same_traj']}same/{r['verify_cross_traj']}cross-traj)  "
            f"K_max: maxlib {r['max_library_size']}/{r['k_max']}")
    Path(args.out).write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
