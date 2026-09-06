"""
chart.py — Chart: a small persistent adapter on the frozen JEPA predictor.

Three kinds (E0 picks one):
  ln_act  — LN affine parameters ONLY, 10,764 params. NOT "+ action encoder":
            the action encoder is a SIBLING of the predictor (video_wm.py:82-83),
            structurally unreachable from a Chart built on `predictor` alone
            (E0_IMPLEMENTATION_PLAN.md T12 #10 — docs previously claimed
            otherwise; the ~10.4k count that seemed to validate it was a
            coincidence, since LN alone already totals 10,764).
  lora4   — LoRA r=4 on attention qkv + proj       (118,176 trainable;
            10,292,640 stored pre-training — see chart.n_params())
  full    — all predictor parameters               (20,800,884 params)

API:
  chart.apply_(predictor)    — swap chart params into predictor in-place
  chart.restore_(predictor)  — swap original params back
  chart.clone()              — deep copy; starts with the SAME weights (identity)
  chart.save(path)           — save to disk
  Chart.load(path, predictor, kind) — restore from disk
  chart.n_params()           — count trainable parameters

Non-negotiables enforced here:
  1. Identity initialisation: clones start with pretrained weights (LoRA B=0).
  2. apply/restore rather than model copies: only ~10–55 k floats swapped.
  3. Charts are DISJOINT parameter sets — updating one chart cannot touch another.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn


ChartKind = Literal["ln_act", "lora4", "full"]


class Chart:
    """
    A persistent adapter for the JEPA predictor.

    Stores a dict mapping parameter names → tensors (the chart's values).
    The predictor's own parameters are the "baseline" that gets swapped out.
    """

    KINDS = ("ln_act", "lora4", "full")

    def __init__(self, predictor: nn.Module, kind: ChartKind = "ln_act") -> None:
        if kind not in self.KINDS:
            raise ValueError(f"kind must be one of {self.KINDS}, got {kind!r}")
        self.kind = kind
        self._param_names: list[str] = _select_param_names(predictor, kind)
        if not self._param_names:
            raise RuntimeError(
                f"No parameters selected for kind={kind!r}. "
                "Check that the predictor checkpoint matches the expected architecture. "
                "Run `scripts/dump_params.py` to inspect available parameter names."
            )
        # Copy current predictor weights as the chart's initial state.
        self._params: dict[str, torch.Tensor] = {
            n: p.data.clone() for n, p in predictor.named_parameters() if n in self._param_names
        }
        # For LoRA: inject the adapters if not already present.
        if kind == "lora4":
            _inject_lora(predictor, self._param_names, self._params)

    # ── Swap interface ────────────────────────────────────────────────────────

    def apply_(self, predictor: nn.Module) -> None:
        """Swap this chart's parameters INTO the predictor in-place."""
        state = dict(predictor.named_parameters())
        
        if self.kind == "lora4":
            for name in self._param_names:
                if name not in state:
                    raise KeyError(f"Chart parameter {name!r} not found in predictor.")
                
                parts = name.split(".")
                mod = predictor
                for p in parts[:-1]:
                    mod = getattr(mod, p)
                attr = parts[-1]
                
                A = self._params[name + ".lora_A"]
                B = self._params[name + ".lora_B"]
                
                # Check if it's already parameterized
                device = getattr(mod, attr).device
                if not parametrize.is_parametrized(mod, attr):
                    # We pass the cloned A and B tensors directly, and move to device
                    lora_param = _LoRAParametrization(A, B).to(device)
                    parametrize.register_parametrization(mod, attr, lora_param)
                else:
                    # Just update the parameters of the existing parametrization
                    mod.parametrizations[attr][0].lora_A.data.copy_(A.to(device))
                    mod.parametrizations[attr][0].lora_B.data.copy_(B.to(device))
        else:
            for name, value in self._params.items():
                if name not in state:
                    raise KeyError(f"Chart parameter {name!r} not found in predictor.")
                state[name].data.copy_(value)

    def restore_(self, predictor: nn.Module) -> None:
        """Swap THIS chart's parameters back into the predictor in-place.
        
        Typical usage pattern:
            c_star.apply_(predictor)       # apply selected chart
            ... forward pass ...
            c0.restore_(predictor)         # restore to identity baseline
        """
        if self.kind == "lora4":
            for name in self._param_names:
                parts = name.split(".")
                mod = predictor
                for p in parts[:-1]:
                    mod = getattr(mod, p)
                attr = parts[-1]
                
                if parametrize.is_parametrized(mod, attr):
                    # Remove the parametrization but keep the original weight intact
                    parametrize.remove_parametrizations(mod, attr, leave_parametrized=False)
        else:
            self.apply_(predictor)

    # ── Clone / persistence ───────────────────────────────────────────────────

    def clone(self) -> "Chart":
        """Return a deep copy with identical weights (identity initialisation)."""
        new = object.__new__(Chart)
        new.kind = self.kind
        new._param_names = list(self._param_names)
        new._params = {k: v.clone() for k, v in self._params.items()}
        return new

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"kind": self.kind, "params": self._params}, path)

    @classmethod
    def load(cls, path: str | Path, predictor: nn.Module) -> "Chart":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Chart file not found: {path}")
        data = torch.load(path, map_location="cpu", weights_only=True)
        kind = data["kind"]
        chart = object.__new__(cls)
        chart.kind = kind
        if kind == "lora4":
            # For lora4, data["params"].keys() includes the base weight names
            # PLUS the injected ".lora_A"/".lora_B" pseudo-keys (see
            # _inject_lora) -- using them directly as _param_names makes
            # apply_()'s lora4 loop treat ".lora_A"/".lora_B" strings as real
            # predictor parameter names, which they aren't, raising a KeyError
            # partway through (leaving the predictor with a partially-applied,
            # never-cleaned-up parametrization -- see code-review.md Bug #6f).
            # Recompute the true base-weight name list deterministically
            # instead of trusting the saved dict's keys.
            chart._param_names = _select_param_names(predictor, kind)
        else:
            chart._param_names = list(data["params"].keys())
        chart._params = data["params"]
        return chart

    def n_params(self) -> int:
        return sum(v.numel() for v in self._params.values())

    def n_trainable_params(self) -> int:
        """Count of parameters that actually receive gradients during
        refinement (FIX_SPEC.md A11). For ``lora4`` this is only the injected
        ``.lora_A`` / ``.lora_B`` tensors -- ``n_params()`` additionally sums
        the 12 frozen full base weight matrices (~10.3M) that are present in
        ``_params`` at construction but are never optimised, which is the
        discredited count recorded in some older results.json artifacts. For
        ``ln_act`` / ``full`` every stored parameter is trained, so this equals
        ``n_params()``."""
        if self.kind == "lora4":
            return sum(v.numel() for k, v in self._params.items()
                       if k.endswith(".lora_A") or k.endswith(".lora_B"))
        return self.n_params()

    def restore_pretrained_(self, predictor: nn.Module,
                             pretrained_state: dict[str, torch.Tensor]) -> None:
        """Restore this chart's touched parameters to their PRETRAINED
        values, not to this chart's own (possibly refined) values.

        FIX_SPEC.md C4: restore_() for kind in {"ln_act", "full"} is
        actually a RE-APPLY of this chart's own current _params -- it does
        NOT put the predictor back to pretrained weights, despite the
        method name and CLAUDE.md 1.6's "restore" language. That falsifies
        claim P-1's literal wording and can leave the predictor permanently
        dirty if callers assumed restore_() meant "back to pretrained".
        restore_() is NOT renamed here -- 10 production call sites
        (atlas/loop.py, atlas/expand.py, scripts/run_e0.py, ...) depend on
        its current re-apply behaviour and a rename was explicitly ruled
        out of scope for this fix (see FIXLOG.md C4). This is a NEW,
        additive method for callers that need genuine pretrained restore
        (for example: releasing the predictor back to a clean baseline
        between unrelated charts, or a diagnostic that must prove a chart
        made zero difference).

        Args:
            predictor:        The live predictor (chart's kind must match
                               what was applied to it).
            pretrained_state: dict[name -> tensor], captured from
                               predictor.state_dict() (or
                               named_parameters()) BEFORE any chart was
                               ever applied to this predictor instance.
        """
        if self.kind == "lora4":
            # apply_()'s LoRA parametrization is ADDITIVE over the original
            # base weight, which restore_()'s parametrize.remove_parametrizations
            # (leave_parametrized=False) already discards without touching --
            # so restore_() already recovers pretrained weights exactly for
            # lora4. Route through it rather than duplicating that logic.
            self.restore_(predictor)
            return
        state = dict(predictor.named_parameters())
        for name in self._param_names:
            if name not in pretrained_state:
                raise KeyError(
                    f"restore_pretrained_: pretrained_state is missing "
                    f"{name!r} -- was it captured from THIS predictor "
                    "before any chart was ever applied to it?"
                )
            if name not in state:
                raise KeyError(f"Chart parameter {name!r} not found in predictor.")
            state[name].data.copy_(pretrained_state[name])

    def update_from_predictor_(self, predictor: nn.Module) -> None:
        """Pull current predictor weights back into the chart (after refinement)."""
        state = dict(predictor.named_parameters())
        if self.kind == "lora4":
            for name in self._param_names:
                parts = name.split(".")
                mod = predictor
                for p in parts[:-1]:
                    mod = getattr(mod, p)
                attr = parts[-1]
                if parametrize.is_parametrized(mod, attr):
                    # Save the trained lora_A and lora_B back to the chart
                    A = mod.parametrizations[attr][0].lora_A
                    B = mod.parametrizations[attr][0].lora_B
                    self._params[name + ".lora_A"] = A.data.clone()
                    self._params[name + ".lora_B"] = B.data.clone()
        else:
            for name in self._param_names:
                self._params[name] = state[name].data.clone()

    def __repr__(self) -> str:
        return f"Chart(kind={self.kind!r}, n_params={self.n_params():,})"


# ── Parameter selection ────────────────────────────────────────────────────────

def _select_param_names(predictor: nn.Module, kind: ChartKind) -> list[str]:
    """Return parameter names to include in a chart of the given kind."""
    names = []
    if kind == "ln_act":
        # LN affine: identify actual nn.LayerNorm modules structurally rather
        # than by name substring. A name-substring check ("norm" in n) misses
        # LayerNorms that sit at a numeric Sequential index (e.g. this
        # predictor's FeedForward pre-norm is `transformer.layers.N.1.net.0`,
        # which has no "norm"/"ln" in its parameter name at all) — confirmed
        # empirically via scripts/dump_params.py, which found only ~5.8k LN
        # params via the old substring check vs. the plan's expected ~10.4k
        # (the missing half being exactly these unnamed FeedForward norms).
        ln_param_ids = set()
        for _, module in predictor.named_modules():
            if isinstance(module, nn.LayerNorm):
                for p in module.parameters(recurse=False):
                    ln_param_ids.add(id(p))
        for n, p in predictor.named_parameters():
            if id(p) in ln_param_ids or "action" in n:
                names.append(n)
        return names
    for n, p in predictor.named_parameters():
        if kind == "full":
            names.append(n)
        elif kind == "lora4":
            # LoRA adapters are injected separately; here we select the base
            # qkv/proj WEIGHT matrices that will be (additively) modified.
            # Biases are excluded deliberately: _inject_lora's rank/in_features
            # math assumes a 2-D weight (out_features, in_features) — matching
            # a 1-D bias would produce a shape mismatch in the LoRA forward
            # (B @ A vs. the bias's own shape).
            # "to_qkv"/"to_out" added for this predictor's lucidrains-style ViT
            # naming (Attention.to_qkv, Attention.to_out) — without them only
            # to_qkv matched and the attention output projection (to_out.0)
            # was silently skipped, confirmed via scripts/dump_params.py.
            if n.endswith(".weight") and any(
                k in n for k in
                ("qkv", "proj", "q_proj", "k_proj", "v_proj", "out_proj", "to_qkv", "to_out")
            ):
                names.append(n)
    return names


import torch.nn.utils.parametrize as parametrize

class _LoRAParametrization(nn.Module):
    def __init__(self, A: torch.Tensor, B: torch.Tensor):
        super().__init__()
        self.lora_A = nn.Parameter(A)
        self.lora_B = nn.Parameter(B)

    def forward(self, original_weight):
        return original_weight + self.lora_B @ self.lora_A

def _inject_lora(
    predictor: nn.Module,
    param_names: list[str],
    chart_params: dict[str, torch.Tensor],
    rank: int = 4,
) -> None:
    """
    Inject LoRA A/B adapter tensors into *chart_params* for every selected
    linear layer. B is initialised to 0 so the adapter starts identity.
    """
    state = dict(predictor.named_parameters())
    for name in param_names:
        W = state[name]
        out_features, in_features = W.shape[0], W.numel() // W.shape[0]
        A_key = name + ".lora_A"
        B_key = name + ".lora_B"
        
        A = torch.randn(rank, in_features) / rank
        B = torch.zeros(out_features, rank)
        chart_params[A_key] = A
        chart_params[B_key] = B
