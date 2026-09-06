"""
scripts/_determinism.py — shared reproducibility setup for Phase-0 diagnostics
and the P0-G collector.

The ~1e-4 forward-scoring drift across process launches (phase0_measure.py run
twice: 48/48 chunks differed) traces to cuBLAS re-autotuning GEMM kernels per
process. This module FIXES the forward path — verified: with it, phase0_measure.py
run twice is bit-identical (0/48 differ).

KNOWN RESIDUAL (do not assume it's gone): the gradient-TRAINING path — chart
fine-tune (`atlas_refine`) and `expand._fit_candidate` — still drifts ~1e-2 in
chart weights across process launches, even with everything below set. A single
backward raises no "non-deterministic algorithm" error (`warn_only=False`), so
it is not a missing-kernel case; the residual is in the CUDA backward
reductions and cannot be removed without patching the vendored jepa-wms
attention. Consequences:
  * forward-only Phase-0 numbers (τ, motion gate, strike rate, σ_r): deterministic.
  * P0-G charts: run once, the output IS the artifact — the residual only means a
    re-run would give slightly different charts, which is acceptable and stated.
  * G7 Group B commit counts: ±1–2 run-to-run — reported as a per-seed
    distribution, never a point estimate.

Import this module BEFORE `import torch` anywhere else (it sets an env var that
torch reads only at CUDA init), then call `make_deterministic(seed)` once after
loading the model.
"""
from __future__ import annotations

import os

# Must be set before the first CUDA context is created — cuBLAS reads it once.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def make_deterministic(seed: int = 0) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Backward through attention: the flash / memory-efficient SDPA kernels have
    # non-deterministic backward on CUDA (the "Memory Efficient attention
    # defaults to a non-deterministic algorithm" warning). Force the math
    # backend, whose backward IS deterministic, for every scaled-dot-product
    # attention call. Costs some speed on the fine-tune backward only — the
    # frozen encoder forward is unaffected in practice.
    for fn in ("enable_flash_sdp", "enable_mem_efficient_sdp", "enable_math_sdp"):
        if hasattr(torch.backends.cuda, fn):
            getattr(torch.backends.cuda, fn)(fn == "enable_math_sdp")
    # warn_only: a few ops (e.g. some scatter/index kernels) have no deterministic
    # implementation; warn rather than raise so a run still completes, and the
    # warning names exactly which op to worry about.
    torch.use_deterministic_algorithms(True, warn_only=True)
    # TF32 off — it is already off on this card, but pin it so a driver/hardware
    # change does not silently reintroduce matmul variance.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def settings_dict(seed: int = 0) -> dict:
    """The exact determinism config, for recording alongside run artifacts —
    so an anomalous chart later can be classified as 'known backward residual'
    vs 'different bug' instead of reflexively blamed on the residual."""
    import subprocess

    import torch

    # P20: on Modal the image `ignore` list drops `.git`, so `git rev-parse`
    # fails and every chart would carry git_commit "unknown". modal_phase0.py
    # reads the SHA locally in its @app.local_entrypoint and passes it through
    # as ATLAS_GIT_SHA.
    git = os.environ.get("ATLAS_GIT_SHA")
    if not git:
        try:
            git = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            git = "unknown"
    try:
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except Exception:
        dirty = None
    return {
        "git_commit": git,
        "git_dirty": dirty,
        "seed": seed,
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "device": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
        "note": "forward-path deterministic; gradient-training path has ~1e-2 "
                "cross-process residual (see module docstring)",
    }
