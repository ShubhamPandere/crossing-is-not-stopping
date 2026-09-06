# ATLAS: Threshold-Crossing vs. Settle-Validated Success

Code and results for an evaluation-methodology study on **Push-T with a frozen JEPA world model**:
threshold-crossing success measures whether a trajectory *transits* the goal set, not whether the
system *stabilizes* in it — under momentum the two diverge, systematically in favor of the worse
controller. All planning runs use a single frozen substrate (DINOv2 encoder + predictor + CEM
planner from [`jepa-wms`](https://github.com/facebookresearch/jepa-wms)); only a small adapter
(≤11k params) is ever trained.

This repository ships only the code that actually produces the paper's reported numbers — traced
directly from `EVIDENCE_LEDGER.md`'s claim-by-claim provenance, not the full experiment codebase.

## Glossary — decode before reading `results/`

Filenames and directory names throughout this repo use these abbreviations consistently:

| Term | Meaning |
|---|---|
| `R0` | Default, unmodified Push-T physics |
| `R1` | High-friction regime (`friction=2.0`) — a prediction-level shift only, not used in the paper's momentum argument |
| `R2` | High-damping regime (`damping=0.5`) — the regime the paper's momentum / threshold-vs-settle argument is built on |
| `dmp` | The literal damping value swept (`dmp01` = damping 0.1, `dmp002` = damping 0.02, etc. — R2 is just "damping=0.5") |
| `nas` | `num_act_stepped` — raw env steps executed before the CEM planner replans (`nas2` ≈ 15 replans/episode, `nas6` ≈ far fewer) |
| `it` | CEM `iterations` per replan (the `it1`/`it3`/`it30` controller-family sweep) |
| `fam` | "Controller family" — the CEM-iteration sweep at fixed adapter state |
| `c2` | The paper's **settle-validated** success criterion (holding position after crossing), vs. the older threshold-crossing pass-through criterion |
| `p0g` / `p0c` | Phase-0 gates G / C — on-policy chart collection / forward-only calibration |
| `ln_act` | The **LayerNorm-affine adapter** — every `nn.LayerNorm`'s weight/bias in the predictor plus any parameter whose name contains "action" (10,764 trainable floats); the only adapter kind reported in the paper |

## Contents

| Directory | Description |
|---|---|
| `atlas/` | Core library — the adapter, UMF scoring, physics regimes, statistics |
| `scripts/collection/` | Drivers that generate raw per-episode data (adapter training, CEM planning, Phase-0 calibration) |
| `scripts/analysis/` | Recomputes every paper statistic from the raw JSONL — one script per reported number |
| `modal/` | Modal launchers for running collection on cloud GPUs |
| `tests/` | Unit tests for `atlas/` |
| `results/figures/` | The 6 camera-ready figures used in the paper (fig0–fig4, figA1) |
| `results/experiment_data/` | Raw per-episode JSON/JSONL outputs backing the paper's numbers, organized by mechanism |

---

## `atlas/`

| File | Description |
|---|---|
| `chart.py` | `Chart` — the LayerNorm-affine (`ln_act`) adapter: apply/restore ~11k floats onto the frozen predictor |
| `score.py` | UMF (unified misprediction fitness) computation, motion-gate calibration |
| `regimes.py` | `PhysicsRegime` — the R0/R1/R2 physics perturbations (friction/damping) |
| `stats.py` | Paired bootstrap CIs, McNemar's test, normalized recovery |
| `harness.py` | Offline chart fine-tuning + per-episode logging, used by `chart_training.py` |
| `router.py` | Scoring baselines (`_e1_score`, `_sdyn_score`) used directly by `phase0_measure.py`; also imported by `harness.py` at load time |
| `library.py` | Imported by `harness.py`/`router.py` at load time; not exercised on the load-bearing call path but kept so those modules import cleanly |
| `expand.py`, `loop.py` | The commit/reject probe and prequential controller used by the Phase-0 G7 calibration scripts (`phase0_g7_group{A,B}.py`) |

## `scripts/collection/`

The drivers that produce the raw data everything else reads.

| File | Description |
|---|---|
| `chart_training.py` | Adapter training / on-policy chart collection driver |
| `cem_planning_eval.py` | CEM-driven planning episodes + UMF scoring for one chart/regime |
| `cem_cost_ranking.py` | Cost-ranking diagnostic — Spearman ρ(CEM cost, true outcome) per adapter, the mechanism behind C1-rank |
| `phase0_measure.py` | Forward-only replay of real Push-T demonstrations under R0/R1/R2 — produces `phase0_chunks.jsonl` |
| `phase0_g7_groupA.py`, `phase0_g7_groupB.py` | Calibration diagnostics behind the motion-gate / adapter-commit thresholds `chart_training.py` uses |
| `merge_p0g_shards.py` | Merges on-policy chart-collection shards from parallel Modal containers |
| `merge_planning_shards.py` | Merges `cem_planning_eval.py` shard outputs, reports UMF-vs-success correlation |
| `download_data.py` | Downloads the `dino_wm_pusht` checkpoint and Push-T dataset |
| `collection_spec.py` | Single source of truth for the collection-protocol CLI flags, shared by the Modal launchers and `tests/test_p0g_guard.py` |
| `determinism.py` | Shared CUDA/cuBLAS determinism setup |

## `scripts/analysis/`

Read-only — every script here recomputes one paper statistic from the raw per-episode JSONL. None import `atlas/`.

⚠️ **Data currency.** On 2026-09-03 a planning-horizon bug (`steps_left`) was found and fixed
(`--fix-steps-left`) in the closed-loop cells feeding the dose-ladder, n=100, and CEM-iteration
results. `damping_dose_ladder.py`, `damping_dose_n100.py`, and `cem_iteration_family.py`'s
PRIMARY iteration ladder have been repointed at the corrected `*_fixsteps` data directories and
re-verified to reproduce the exact numbers in the project's audit trail (Δ-25.6px / 60/100 for
the n=100 headline, 0.95/0.75/0.70/0.65/0.55/0.50 crossing-SR for the dose ladder, 59.0/122.5/
154.5/149.8px for the iteration ladder). Each writes a `*_fixsteps.json` output, kept alongside
(not overwriting) the original archived JSON of the same vintage already in `results/` — the
archived files are retained for the historical record only and should not be quoted as current.
`cem_iteration_family.py`'s SECONDARY 12-controller matrix has no corrected rerun available for
its ln_act/alpha0/nas6 cells and remains on the original data (see the script's own docstring).

| File | Produces |
|---|---|
| `settle_threshold_sweep.py` | The settle-validated-vs-threshold-crossing sweep over acceptance thresholds |
| `offline_umf_analysis.py` | Widens the acceptance-criterion UMF measurement from 40 held-out windows to every on-policy chunk |
| `damping_dose_ladder.py` | The damping dose-response ladder — pass-through vs. settled success rate vs. damping (corrected, writes `damping_dose_ladder_fixsteps.json`) |
| `damping_dose_n50.py` | N=50 replication of the dose-ladder result on a disjoint task set |
| `damping_dose_n100.py` | N=100 headline — three disjoint task sets + merged, **the paper's §4 lead statistic** (corrected, writes `damping_dose_n100_fixsteps.json`) |
| `damping_02_transfer.py` | d=0.2 adapter-transfer + d=0.1 consistency rerun, reusing `damping_dose_n100.py`'s exact statistical convention |
| `cem_iteration_family.py` | The 12-controller CEM-iteration family sweep (rank-inversion result); PRIMARY table corrected, writes `cem_iteration_family_fixsteps.json` |
| `coast_phase_model.py` | Analytical residual-momentum ("coast") model validation |
| `coast_window_robustness.py` | Robustness check on the coast-model's window choice |
| `statistical_power_screen.py` | Acceptance-screen power/discrimination table |
| `ladder_statistical_inference.py` | Per-cell McNemar + paired-bootstrap CI and a monotonic-trend test across the dose ladder |
| `keepplanning_vs_replan.py` | The keep-planning control: does continuous replanning through the hold change the result? (paper Appendix, `app:keepplan`) |
| `make_paper_figures.py` | Builds all 6 camera-ready figures (`results/figures/`) from the raw data below |

## `modal/`

Cloud-GPU launchers — see [`modal/README.md`](modal/README.md) for setup and usage.

| File | Description |
|---|---|
| `modal_chart_collection.py` | Phase-0 calibration + on-policy chart collection |
| `modal_cem_planning.py` | Adapter training + CEM planning-success evaluation |

## `tests/`

| File | Description |
|---|---|
| `test_score.py` | Unit tests for UMF computation (`atlas.score`) |
| `test_stats.py` | Unit tests for statistics utilities (`atlas.stats`) |
| `test_p0g_guard.py` | Regression guard: the Modal launcher and the local collector must emit identical collection-defining flags |

## `results/figures/`

The 6 figures actually used in the paper, built by `scripts/analysis/make_paper_figures.py`.

| File | Used for |
|---|---|
| `fig0_example.pdf` | One example episode illustrating threshold-crossing vs. settling |
| `fig1_dose_ladder.pdf` | The damping dose-response ladder (§3) |
| `fig2_drift.pdf` | Block drift over the post-crossing hold at three damping values (§3) |
| `fig3_iteration_ladder.pdf` | The CEM-iteration rank-inversion result (§4) |
| `fig4_coast.pdf` | Predicted-vs-measured residual-momentum ("coast") model validation (§5) |
| `figA1_controller_scatter.pdf` | Appendix: the 12-controller scatter (deliberately uses the archived, pre-`--fix-steps-left` data — see the script's docstring) |

## `results/experiment_data/`

Raw per-episode records and calibration logs underlying the paper's tables — JSON/JSONL/TXT
only, no trained weights. Organized into 5 subfolders by mechanism (not the flat, ~230-entry
dump this repo shipped with earlier — decode filenames using the glossary above):

| Subfolder | Backs | Contents |
|---|---|---|
| `dose_ladder/` | B3-dose-ladder(-fixed), N13-screen-power, N17-keepplanning | The damping sweep (`ladder_dmp*`, `smoke_dmp002_*`), the central d=0.5 cells (`c2_settle2_*`), the keep-planning control (`keepplan_*`), `c2_p0g_R2`/`p0c` (C2-head) |
| `scaling_replication/` | N12-n50, N15-n100 | `n50_*`, `n100_*` — the adapter-vs-frozen comparison at increasing sample size |
| `iteration_ladder/` | N16-fixed-ladder | `fam_it*`, `fam_alpha0_*` — the CEM-iteration controller-family sweep |
| `transfer_and_routing/` | B2-transfer-01, N18, N19, C1-rank, C2-route | `dmp0{1,2}_transfer_*`, `p0g_onpolicy*`, `cost_ranking_*`, the `d02`/`d05`/`r0` offline-UMF cross-checks |
| `calibration/` | Phase-0 background (τ, motion gate, G4/G7) | `g4_*`, `g7_*`, `phase0_chunks.jsonl`, `phase0_summary.json`, `p0c` |

Each subfolder also holds the corresponding analysis script's JSON output (e.g.
`dose_ladder/damping_dose_ladder_fixsteps.json`). Note: `scripts/analysis/*.py` read from a
flat `phase0_v3/` working directory by convention (see "Reproducing results" below) — this
subfolder layout is for readability when browsing the shipped snapshot, not the scripts' own
runtime path.

---

## Installation

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -e ".[dev]"

# jepa-wms substrate (frozen encoder + predictor + CEM planner)
git clone https://github.com/facebookresearch/jepa-wms vendor/jepa-wms
pip install -e vendor/jepa-wms

atlas-download   # fetch the dino_wm_pusht checkpoint + Push-T dataset
```

## Quick start

```bash
pytest tests/
```

## Reproducing results

```bash
# 1. Collect Phase-0 calibration data + train the adapter
python scripts/collection/phase0_measure.py
atlas-chart-training --kind ln_act --regime R2

# 2. Run CEM planning episodes
atlas-cem-planning --kind ln_act --regime R2

# 3. Recompute a paper statistic from the resulting JSONL
python scripts/analysis/damping_dose_n100.py
python scripts/analysis/cem_iteration_family.py
# ... etc — see the scripts/analysis/ table above for what each produces

# 4. Rebuild the paper's figures from the analysis output
python scripts/analysis/make_paper_figures.py
```

## Citation

```bibtex
@inproceedings{atlas2026,
  title     = {Threshold-Crossing vs. Settle-Validated Success in Continual World Models},
  author    = {Pandere, Shubham},
  booktitle = {NeurIPS 2026 Workshop on World Models in Physical AI},
  year      = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
