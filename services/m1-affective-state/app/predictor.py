"""Module 1 predictor.

BASELINE (v0.1.0): transparent rules over server-extracted features → softmax over the five
AffectiveLabel classes. It exists so the full app pipeline works from day one.

REPLACE with the trained model: keep the class name, `info` and `predict()` signature; load the model
from `models/` in `__init__`, call `m1_affect.features.extract_all()` and feed the features to it.
"""

from __future__ import annotations

import math

from m1_affect.features import extract_all
from wellness_contracts import AffectiveLabel, AffectiveStateRequest, AffectiveStateResult, ModelInfo

# Reference values for a resting adult using a phone normally (used to standardise features).
REF = {
    "hr_bpm": (72.0, 12.0),  # mean, scale
    "hrv_rmssd_ms": (40.0, 15.0),
    "backspace_rate": (0.06, 0.04),
    "typing_speed_kpm": (150.0, 50.0),
    "unlocks_per_hour": (4.0, 3.0),
    "app_switches_per_hour": (10.0, 6.0),
    "steps_per_hour": (300.0, 300.0),
    "social_share": (0.3, 0.2),
}


def _z(features: dict, key: str) -> float:
    """Standardised deviation; 0 when the feature is missing (no evidence either way). Clipped to ±3."""
    v = features.get(key)
    if v is None:
        return 0.0
    mean, scale = REF[key]
    return max(-3.0, min(3.0, (v - mean) / scale))


def rule_scores(f: dict) -> dict[AffectiveLabel, float]:
    hr, hrv_low = _z(f, "hr_bpm"), -_z(f, "hrv_rmssd_ms")
    err, speed = _z(f, "backspace_rate"), _z(f, "typing_speed_kpm")
    unlock, switch = _z(f, "unlocks_per_hour"), _z(f, "app_switches_per_hour")
    move, social = _z(f, "steps_per_hour"), _z(f, "social_share")
    return {
        # high arousal, negative valence: fast heart, low HRV, typing errors, fragmented phone use
        AffectiveLabel.stress: 0.9 * hr + 0.8 * hrv_low + 0.6 * err + 0.4 * switch + 0.3 * unlock,
        # high arousal, restless checking behaviour
        AffectiveLabel.anxiety: 0.6 * hr + 0.5 * hrv_low + 0.7 * unlock + 0.6 * switch + 0.3 * err,
        # low arousal, calm physiology and steady interaction
        AffectiveLabel.relaxed: 0.2 - 0.7 * hr - 0.6 * hrv_low - 0.4 * err - 0.4 * switch,
        # positive valence: movement, fluent typing
        AffectiveLabel.happy: 0.1 + 0.5 * move + 0.3 * speed - 0.4 * err - 0.3 * hrv_low,
        # low valence, low activation: little movement, slow typing, passive consumption
        AffectiveLabel.sad: 0.1 - 0.6 * move - 0.4 * speed + 0.3 * social - 0.2 * unlock,
    }


def softmax(scores: dict[AffectiveLabel, float]) -> dict[AffectiveLabel, float]:
    m = max(scores.values())
    e = {k: math.exp(v - m) for k, v in scores.items()}
    s = sum(e.values())
    return {k: round(v / s, 6) for k, v in e.items()}


class Predictor:
    info = ModelInfo(name="m1-rule-baseline", version="0.1.0")

    def predict(self, req: AffectiveStateRequest) -> AffectiveStateResult:
        features = extract_all(req)
        probs = softmax(rule_scores(features))
        label = max(probs, key=lambda k: (probs[k], k.value))
        return AffectiveStateResult(
            user_id=req.user_id,
            window_start=req.window_start,
            window_end=req.window_end,
            label=label,
            probabilities=probs,
            confidence=probs[label],
            features={k: (round(v, 4) if isinstance(v, float) else v) for k, v in features.items()},
            model=self.info,
        )
