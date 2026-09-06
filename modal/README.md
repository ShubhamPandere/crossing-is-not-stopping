# Modal — ATLAS Contributor Guide

## Prerequisites

```bash
pip install modal
modal token new   # authenticate with your Modal account
```

## First run: create the persistent volume

```bash
modal volume create atlas-data
```

This volume persists all checkpoints, data, logs, and outputs across runs.

## Running experiments

Each launcher below is a standalone Modal app (`@app.local_entrypoint`), not a
shared `modal_app.py` — invoke each file directly.

```bash
# Phase-0 calibration + on-policy chart collection
modal run modal/modal_chart_collection.py::main
modal run modal/modal_chart_collection.py::p0g-collect --regime R2

# Adapter training + CEM planning evaluation
modal run modal/modal_cem_planning.py::main --kind ln_act --regime R2
```

Run `modal run modal/<file>.py --help` for the full flag list on any launcher.

## Downloading results

```bash
modal volume get atlas-data /atlas_root/atlas_out ./atlas_out_from_modal
```

## GPU selection

GPU type is pinned per-function via `@app.function(gpu=...)` inside each
launcher (e.g. `modal_chart_collection.py` uses `T4`/`L4`). Edit the decorator to change it.

## Env vars inside the container

All ATLAS paths are pre-set via the image's `.env()` call in each launcher:

| Variable        | Value in container               |
|-----------------|----------------------------------|
| `ATLAS_HOME`    | `/atlas_root`                    |
| `JEPAWM_DSET`   | `/atlas_root/data`               |
| `JEPAWM_CKPT`   | `/atlas_root/ckpts`              |
| `ATLAS_OUT`     | `/atlas_root/atlas_out`          |
| `TORCH_HOME`    | `/atlas_root/hub`                |
| `MUJOCO_GL`     | `egl`                            |
