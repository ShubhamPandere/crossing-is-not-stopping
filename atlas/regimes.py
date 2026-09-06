"""
regimes.py — Physics regime wrappers and visual corruption wrappers for Push-T.

Three dynamics regimes (primary, matched appearance) — see REGIME_CONFIGS below,
which is authoritative:
  R0  default — shipped parameters (friction=0.0, elasticity=0.0, damping=0)
  R1  high friction — agent+block shape.friction = 2.0 (was: mass x0.2, then 0.8)
  R2  high damping — space.damping = 0.5 (was: mass x0.2 → elasticity 0.9, both
      proven mechanically dead: space.damping is hardcoded 0 in the shipped env,
      which annihilates restitution velocity; damping is the axis that survives.
      P21: this line previously read "shape.elasticity raised", contradicting
      REGIME_CONFIGS 60 lines down — corrected 2026-08-29.)

CORRECTNESS-CRITICAL BUG FOUND 2026-08-23, see REGIME_DESIGN_REVIEW.md and
ATLAS_implementation_plan_v2.md §6.1a: R1 was originally specified as a T-block
mass x0.2 scaling. Empirically and analytically confirmed dead — Push-T's pusher
agent is a `pymunk.Body(body_type=pymunk.Body.KINEMATIC)`, and Chipmunk2D's
impulse solver treats a kinematic body as having infinite mass, which makes the
post-collision velocity of the body it hits algebraically INDEPENDENT of that
body's own mass/moment, at any scale (0.001x to 1000x tested, byte-identical
trajectories every time). No mass value can ever produce a regime shift here.
Re-targeted onto shape.friction (R1) and space.damping (R2) instead — friction
is a dimensionless coefficient, damping a global velocity-retention rate; neither
is a mass-like quantity, so neither is subject to the cancellation above
(confirmed empirically, see below).

NON-OBVIOUS TRAP: pymunk combines two colliding shapes' friction/elasticity
multiplicatively-ish (empirically: if EITHER shape's friction/elasticity is
0.0, the combined effect is 0.0 regardless of the other shape's value). Since
the agent's shape defaults to friction=elasticity=0.0 same as the block's,
_apply_physics() must set both the AGENT's shapes and the BLOCK's shapes —
setting only the block (the naive choice) has no effect at all.

Visual corruptions (E2 only — appearance differs, dynamics unchanged):
  blur, salt_pepper, dark, colour_change
  Ported from AdaJEPA (github.com/agentic-learning-ai-lab/adajepa).

CRITICAL: _apply_physics() is called inside reset() because many pymunk envs
rebuild the space on reset, which resets physics parameters.
"""

from __future__ import annotations

from typing import Literal

# NOT gymnasium: PushTEnv (evals/simu_env_planning/envs/pusht_env/pusht_env.py)
# and jepa-wms's own PushTWrapper both subclass the legacy `gym` package
# (0.23.1) — gymnasium.Wrapper.__init__ asserts isinstance(env, gymnasium.Env)
# and raises immediately on a legacy-gym env, so this module never actually
# worked against the real Push-T environment until this was corrected.
import gym
import numpy as np

RegimeName = Literal["R0", "R1", "R2"]

# Physics parameter values. R1/R2 re-targeted 2026-08-23 (mass_scale/damping
# proven non-functional against a kinematic pusher — see module docstring).
# Values chosen well above the shipped 0.0 baseline for both parameters, and
# empirically confirmed (test_physicsregime-style direct repro) to produce a
# visible collision-response difference once contact occurs.
# FUTURE "R3" candidate (not implemented) — agent PD tracking gains k_p/k_v
# (pusht_env.py:384, = 100, 20). The only appearance-matched dynamics lever left
# unused: friction (R1) is prediction-level only and saturates at 2.0, elasticity
# is mechanically dead (space.damping hardcoded 0), mass cancels, action_scale /
# geometry are confounded / not matched. Changing k_p/k_v changes the KINEMATIC
# pusher's own motion -> v_rel at contact -> a real, appearance-matched dynamics
# shift with both prediction- and trajectory-level effects. Needs a _get_agent()
# PD-gain setter + its own G4 calibration before use. See
# research_audit/IMPLEMENTATION_PLAN_V3.md §15 item 6 and REGIME_DESIGN_REVIEW.md.
REGIME_CONFIGS: dict[str, dict] = {
    "R0": {},                              # default: no modifications
    # Calibrated 2026-08-25 (E0_RECOVERY_PLAN.md P0/P1, see §0.3-0.5). These
    # are the DEFAULTS so nothing downstream depends on remembering a
    # --regime-config override flag.
    "R1": {"friction": 2.0},              # friction saturates here (5.0 measured identical); was 0.8
    "R2": {"damping": 0.5},               # was elasticity 0.9 -- proven mechanically DEAD (§0.3):
                                           # space.damping=0 in the shipped env annihilates all
                                           # restitution velocity before it can displace anything.
                                           # Re-targeted onto space.damping, the axis that actually
                                           # expresses itself (§0.3-0.4).
}


def set_regime_config(name: str, cfg: dict) -> None:
    """Override a regime's physics parameters for calibration sweeps (P1).
    Values are logged per-run; this is a benchmark-design knob, not a tuned
    hyperparameter (see REGIME_CALIBRATION.md)."""
    REGIME_CONFIGS[name] = dict(cfg)


class PhysicsRegime(gym.Wrapper):
    """
    Gymnasium wrapper that applies physics modifications to a pymunk-based env.

    Args:
        env:          The base Push-T environment.
        regime:       One of 'R0', 'R1', 'R2'.

    Raises:
        ValueError:   If regime is not recognised.
        RuntimeError: If the expected pymunk attributes are not found on the env
                      (catches wrong env or changed internals early).
    """

    def __init__(self, env: gym.Env, regime: RegimeName) -> None:
        super().__init__(env)
        if regime not in REGIME_CONFIGS:
            raise ValueError(
                f"Unknown regime {regime!r}. Valid options: {list(REGIME_CONFIGS.keys())}"
            )
        self.regime = regime
        self._cfg = REGIME_CONFIGS[regime]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._apply_physics()
        return obs, info

    def _apply_physics(self) -> None:
        """Apply physics modifications. Called after every reset."""
        if not self._cfg:
            return  # R0: no modifications

        if "friction" in self._cfg:
            self._set_shape_property("friction", self._cfg["friction"])

        if "elasticity" in self._cfg:
            self._set_shape_property("elasticity", self._cfg["elasticity"])

        if "damping" in self._cfg:
            # space.damping is fraction of velocity retained PER SECOND (pymunk),
            # hardcoded to 0 in PushTEnv._setup() (called by every reset(), which
            # runs before this) -- so setting it here, after reset(), is both
            # necessary (else it's clobbered back to 0) and correctly ordered.
            # gym.Wrapper proxies attribute READS, so self.env.space resolves to
            # the freshly-rebuilt space; the pusher is KINEMATIC (velocity set
            # explicitly by PD control each sim step), so this affects the BLOCK
            # only -- see E0_RECOVERY_PLAN.md §0.3.
            self.env.space.damping = self._cfg["damping"]

    def _set_shape_property(self, name: str, value: float) -> None:
        """Set a pymunk Shape property (friction/elasticity) on BOTH the agent's
        and the block's shapes. Both must be set: pymunk combines two colliding
        shapes' friction/elasticity such that if EITHER is 0.0 (the shipped
        default for both agent and block here), the combined effect is 0.0
        regardless of the other shape's value — confirmed empirically. Setting
        only the block (the naive choice) silently produces no regime shift."""
        for body in (self._get_agent(), self._get_block()):
            for shape in body.shapes:
                setattr(shape, name, value)

    def _get_agent(self):
        """Locate the pusher agent body."""
        for attr_path in ("env.agent", "agent", "env.env.agent", "unwrapped.agent"):
            obj = self.env
            try:
                for attr in attr_path.split("."):
                    obj = getattr(obj, attr)
                return obj
            except AttributeError:
                continue
        raise RuntimeError(
            "Could not locate agent body on the environment. "
            "Check the env's attribute structure with `vars(env.unwrapped)` and "
            "update _get_agent() in regimes.py."
        )

    def _get_block(self):
        """Locate the T-block body."""
        for attr_path in ("env.block", "block", "env.env.block",
                          "unwrapped.block", "env.tee", "unwrapped.tee"):
            obj = self.env
            try:
                for attr in attr_path.split("."):
                    obj = getattr(obj, attr)
                return obj
            except AttributeError:
                continue
        raise RuntimeError(
            "Could not locate T-block body on the environment. "
            "Check the env's attribute structure with `vars(env.unwrapped)` and "
            "update _get_block() in regimes.py."
        )


# ── Visual corruptions (E2 only) ───────────────────────────────────────────────

CorruptionKind = Literal["blur", "salt_pepper", "dark", "colour_change", "none"]


class VisualCorruption(gym.ObservationWrapper):
    """
    Observation wrapper that applies a fixed visual corruption while leaving
    physics completely unchanged (appearance differs, dynamics same — E2 Cell C/D).

    Args:
        env:       Base environment.
        kind:      Type of corruption to apply.
        severity:  Strength parameter (interpretation depends on kind).
    """

    def __init__(self, env: gym.Env, kind: CorruptionKind, severity: float = 0.5,
                 seed: int = 0) -> None:
        super().__init__(env)
        if kind not in ("blur", "salt_pepper", "dark", "colour_change", "none"):
            raise ValueError(f"Unknown corruption kind: {kind!r}")
        self.kind = kind
        self.severity = severity
        # Own RNG rather than a fresh default_rng() per frame: salt_pepper is
        # otherwise non-reproducible, which silently breaks G5's paired-seed
        # guarantee (two arms on the same seed would get different noise).
        self._rng = np.random.default_rng(seed)

    def reset(self, **kwargs):
        # PushTEnv.reset() returns (obs, state), not a bare obs -- legacy
        # gym.ObservationWrapper.reset() would hand that whole TUPLE to
        # observation(). Same wrapper-API mismatch PhysicsRegime had.
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple):
            return (self.observation(out[0]),) + tuple(out[1:])
        return self.observation(out)

    def observation(self, obs):
        if self.kind == "none":
            return obs
        # PushTEnv's obs is a dict {"visual": [H,W,3] uint8, "proprio": [...]}.
        # Corrupt ONLY the image: proprio and physics must stay untouched, which
        # is exactly what makes E2's Cell C an appearance-only shift.
        if isinstance(obs, dict):
            if "visual" not in obs:
                return obs
            out = dict(obs)
            out["visual"] = _corrupt(np.asarray(out["visual"]), self.kind,
                                     self.severity, self._rng)
            return out
        obs = np.asarray(obs)
        if obs.ndim == 1:
            # Flattened; return as-is (not an image).
            return obs
        return _corrupt(obs, self.kind, self.severity, self._rng)


def _corrupt(img: np.ndarray, kind: CorruptionKind, severity: float,
             rng: "np.random.Generator | None" = None) -> np.ndarray:
    """Apply a visual corruption to a uint8 image array [..., H, W, C] or [H, W, C]."""
    import cv2

    img = img.copy()
    if kind == "blur":
        ksize = max(1, int(severity * 20) | 1)  # must be odd
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    elif kind == "salt_pepper":
        rng = np.random.default_rng() if rng is None else rng
        mask = rng.random(img.shape[:2])
        img[mask < severity / 2] = 0
        img[mask > 1 - severity / 2] = 255

    elif kind == "dark":
        img = (img * (1.0 - severity * 0.8)).clip(0, 255).astype(np.uint8)

    elif kind == "colour_change":
        # Hue shift via HSV.
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + int(severity * 90)) % 180
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    return img


def assert_no_bypassed_corruption(regime_or_env: gym.Env) -> None:
    """FIX_SPEC.md C5: several call sites (atlas/harness.py, atlas/
    harness_e4.py, scripts/run_e0_planning.py, scripts/profile_episode.py,
    scripts/diagnose_cem_costs.py) step `base_env` directly rather than the
    (possibly wrapped) `env`/`regime` object, for planner-loop reasons
    unrelated to this fix. That is harmless for PhysicsRegime (a
    non-observation-mutating gym.Wrapper -- it only touches physics
    parameters in reset(), never observations, so bypassing it in step()
    changes nothing observable) but FATAL AND SILENT for VisualCorruption:
    its corruption only ever applies inside VisualCorruption.observation(),
    which base_env.step() never reaches, so a corrupted planning run driven
    that way would silently score/execute against the CLEAN observation
    while believing it corrupted -- exactly this project's E2 Cell C/D
    appearance-only-shift design.

    Call this once, at episode-loop entry, on whatever `regime`/`env`
    object the caller was ABOUT to bypass by stepping `base_env` directly.
    Raises loudly instead of silently no-op-ing the corruption.
    """
    obj = regime_or_env
    seen = 0
    while isinstance(obj, gym.Wrapper):
        if isinstance(obj, VisualCorruption) and obj.kind != "none":
            raise RuntimeError(
                "assert_no_bypassed_corruption: a VisualCorruption wrapper "
                f"(kind={obj.kind!r}) wraps this episode's env, but this "
                "call site steps base_env directly, which skips "
                "VisualCorruption.observation() entirely -- the corruption "
                "would never be applied. Step the wrapper chain (call "
                ".step() on the outermost wrapper), not base_env, for any "
                "episode that uses a visual corruption."
            )
        obj = obj.env
        seen += 1
        if seen > 10:
            break  # defensive: malformed wrapper chain, not this function's job


def build_env(
    base_env: gym.Env,
    regime: RegimeName = "R0",
    corruption: CorruptionKind = "none",
    corruption_severity: float = 0.5,
    corruption_seed: int = 0,
) -> gym.Env:
    """
    Compose a Push-T environment with the specified regime and visual corruption.

    Args:
        base_env:             The raw Push-T gym environment (already instantiated).
        regime:               Physics regime to apply.
        corruption:           Visual corruption (E2 only; use 'none' otherwise).
        corruption_severity:  Severity parameter for the corruption.

    Returns:
        Wrapped environment ready for episodes.
    """
    env = PhysicsRegime(base_env, regime)
    if corruption != "none":
        env = VisualCorruption(env, corruption, corruption_severity, seed=corruption_seed)
    return env
