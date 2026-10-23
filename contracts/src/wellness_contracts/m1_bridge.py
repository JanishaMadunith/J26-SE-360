"""Bridge between M1 (Multimodal Affective State Detection) and M2 (Cognitive Wellness Prediction)."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any
import uuid

import numpy as np

from .common import AffectiveLabel, ModelInfo
from .modules import (
    AffectiveStateResult,
    ManualWellnessDay,
)

# M1 classes used in the trained models and prototype
M1_CLASSES = ["Neutral", "Calm", "Happy", "Sad", "Angry"]

# Mapping from M1 classes to AffectiveLabel in contracts
M1_TO_CONTRACT_LABEL = {
    "Neutral": AffectiveLabel.relaxed,  # baseline equilibrium
    "Calm": AffectiveLabel.relaxed,
    "Happy": AffectiveLabel.happy,
    "Sad": AffectiveLabel.sad,
    "Angry": AffectiveLabel.stress,
}


def map_m1_to_current_wellness(
    fused_probs: list[float] | np.ndarray,
    typing_metrics: dict[str, float] | None = None,
) -> dict[str, float]:
    """Map M1's 5-class fused emotion probabilities and typing biometrics to M2's 5 cognitive/wellness inputs:

    - stress
    - overwhelmed
    - mental_tiredness
    - energy
    - focus

    fused_probs: [P(Neutral), P(Calm), P(Happy), P(Sad), P(Angry)]
    """
    probs = np.array(fused_probs, dtype=float)
    if len(probs) != 5:
        raise ValueError(f"Expected 5 class probabilities for [Neutral, Calm, Happy, Sad, Angry], got {len(probs)}")

    p_neutral, p_calm, p_happy, p_sad, p_angry = probs
    typing = typing_metrics or {}

    wpm = typing.get("wpm", 42.0)
    avg_hold = typing.get("hold_time", 90.0)
    jitter = typing.get("jitter", 35.0)
    error_rate = typing.get("error_rate", 4.0)

    # Motor-cadence fatigue penalty from typing (lingering key hold > 120ms or sluggish wpm < 30)
    typing_fatigue = 0.0
    if avg_hold > 120:
        typing_fatigue += min(0.20, (avg_hold - 120) / 200.0)
    if wpm < 30:
        typing_fatigue += min(0.15, (30 - wpm) / 40.0)

    # Motor distraction penalty (high jitter or frequent error backspaces)
    typing_distraction = 0.0
    if jitter > 55:
        typing_distraction += min(0.20, (jitter - 55) / 80.0)
    if error_rate > 10:
        typing_distraction += min(0.20, (error_rate - 10) / 30.0)

    # 1. Stress & Overwhelmed: Driven by Angry/Agitation + portion of distress/Sad
    stress = float(np.clip(p_angry * 0.95 + p_sad * 0.25 + typing_distraction * 0.5, 0.0, 1.0))
    overwhelmed = float(np.clip(p_angry * 1.05 + p_sad * 0.20 + typing_distraction * 0.4, 0.0, 1.0))

    # 2. Mental Tiredness: Driven by Sadness/sluggishness + typing fatigue
    mental_tiredness = float(np.clip(p_sad * 0.70 + p_angry * 0.20 + typing_fatigue, 0.0, 1.0))

    # 3. Energy: High with Happy & Calm, reduced by Sad, Angry, and fatigue
    energy = float(
        np.clip(
            0.50 + 0.45 * p_happy + 0.15 * p_calm - 0.35 * p_sad - 0.25 * p_angry - 0.5 * typing_fatigue,
            0.0,
            1.0,
        )
    )

    # 4. Focus: High with Calm rhythm & Happy, degraded by Angry, high jitter, and error corrections
    focus = float(
        np.clip(
            0.65 * p_calm + 0.30 * p_happy + 0.10 * p_neutral - 0.40 * p_angry - typing_distraction,
            0.0,
            1.0,
        )
    )

    return {
        "stress": round(stress, 4),
        "overwhelmed": round(overwhelmed, 4),
        "mental_tiredness": round(mental_tiredness, 4),
        "energy": round(energy, 4),
        "focus": round(focus, 4),
    }


def m1_fused_to_affective_state_result(
    user_id: uuid.UUID | str,
    window_start: datetime,
    window_end: datetime,
    fused_emotion: str,
    fused_confidence: float,
    fused_probs: list[float] | np.ndarray,
    features: dict[str, Any] | None = None,
) -> AffectiveStateResult:
    """Package M1 fused output into the standard AffectiveStateResult contract."""
    probs = np.array(fused_probs, dtype=float)
    contract_label = M1_TO_CONTRACT_LABEL.get(fused_emotion, AffectiveLabel.stress)

    # Map the 5 classes into standard AffectiveLabel probabilities
    prob_stress = float(probs[4])
    prob_relaxed = float(probs[1] + 0.5 * probs[0])
    prob_happy = float(probs[2])
    prob_sad = float(probs[3])
    prob_anxiety = float(0.5 * probs[4] + 0.1 * probs[0])

    raw_dict = {
        AffectiveLabel.stress: prob_stress,
        AffectiveLabel.relaxed: prob_relaxed,
        AffectiveLabel.happy: prob_happy,
        AffectiveLabel.sad: prob_sad,
        AffectiveLabel.anxiety: prob_anxiety,
    }
    s = sum(raw_dict.values()) or 1.0
    probabilities = {k: round(v / s, 4) for k, v in raw_dict.items()}

    uid = user_id if isinstance(user_id, uuid.UUID) else uuid.UUID(str(user_id))
    return AffectiveStateResult(
        user_id=uid,
        window_start=window_start,
        window_end=window_end,
        label=contract_label,
        probabilities=probabilities,
        confidence=round(float(fused_confidence), 4),
        features=features or {},
        model=ModelInfo(name="m1-multimodal-late-fusion", version="1.0.0"),
    )


def create_current_manual_wellness_day(
    target_date: date,
    fused_probs: list[float] | np.ndarray,
    typing_metrics: dict[str, float] | None = None,
    sleep_hours: float = 7.0,
    wake_hour: float = 7.0,
    steps: int = 5000,
    tasks_due_24h: int = 2,
    minutes_due_24h: int = 120,
    tasks_overdue: int = 0,
    tasks_pending: int = 2,
    bedtime_hour: float = 23.0,
    next_wake_hour: float = 7.0,
) -> ManualWellnessDay:
    """Create a ManualWellnessDay for the current day seeded with M1 fused outputs."""
    derived = map_m1_to_current_wellness(fused_probs, typing_metrics)
    return ManualWellnessDay(
        date=target_date,
        sleep_hours=sleep_hours,
        wake_hour=wake_hour,
        steps=steps,
        stress=derived["stress"],
        mental_tiredness=derived["mental_tiredness"],
        focus=derived["focus"],
        energy=derived["energy"],
        overwhelmed=derived["overwhelmed"],
        bedtime_hour=bedtime_hour,
        next_wake_hour=next_wake_hour,
        tasks_due_24h=tasks_due_24h,
        minutes_due_24h=minutes_due_24h,
        tasks_overdue=tasks_overdue,
        tasks_pending=tasks_pending,
    )


def get_preset_history_scenarios(reference_date: date) -> dict[str, list[ManualWellnessDay]]:
    """Generate calibrated 3-day history scenarios for past days (Day -3, -2, -1)."""
    d1 = reference_date - timedelta(days=1)
    d2 = reference_date - timedelta(days=2)
    d3 = reference_date - timedelta(days=3)

    return {
        "Balanced Normal Week": [
            ManualWellnessDay(
                date=d3,
                sleep_hours=7.5,
                wake_hour=7.0,
                steps=7200,
                stress=0.25,
                mental_tiredness=0.30,
                focus=0.70,
                energy=0.75,
                overwhelmed=0.20,
                tasks_due_24h=2,
                minutes_due_24h=90,
                tasks_overdue=0,
                tasks_pending=2,
            ),
            ManualWellnessDay(
                date=d2,
                sleep_hours=7.0,
                wake_hour=7.0,
                steps=6800,
                stress=0.30,
                mental_tiredness=0.35,
                focus=0.65,
                energy=0.70,
                overwhelmed=0.25,
                tasks_due_24h=3,
                minutes_due_24h=120,
                tasks_overdue=0,
                tasks_pending=3,
            ),
            ManualWellnessDay(
                date=d1,
                sleep_hours=7.2,
                wake_hour=7.0,
                steps=6100,
                stress=0.28,
                mental_tiredness=0.32,
                focus=0.68,
                energy=0.72,
                overwhelmed=0.22,
                tasks_due_24h=2,
                minutes_due_24h=100,
                tasks_overdue=0,
                tasks_pending=2,
            ),
        ],
        "Exam / Crunch Week (High Stress)": [
            ManualWellnessDay(
                date=d3,
                sleep_hours=6.0,
                wake_hour=6.5,
                steps=4200,
                stress=0.65,
                mental_tiredness=0.60,
                focus=0.50,
                energy=0.45,
                overwhelmed=0.60,
                tasks_due_24h=5,
                minutes_due_24h=240,
                tasks_overdue=1,
                tasks_pending=6,
            ),
            ManualWellnessDay(
                date=d2,
                sleep_hours=5.5,
                wake_hour=6.0,
                steps=3800,
                stress=0.75,
                mental_tiredness=0.70,
                focus=0.45,
                energy=0.40,
                overwhelmed=0.75,
                tasks_due_24h=6,
                minutes_due_24h=300,
                tasks_overdue=2,
                tasks_pending=7,
            ),
            ManualWellnessDay(
                date=d1,
                sleep_hours=5.0,
                wake_hour=6.0,
                steps=3200,
                stress=0.85,
                mental_tiredness=0.80,
                focus=0.35,
                energy=0.30,
                overwhelmed=0.85,
                tasks_due_24h=7,
                minutes_due_24h=360,
                tasks_overdue=3,
                tasks_pending=8,
            ),
        ],
        "Post-Weekend Recovery (Low Stress)": [
            ManualWellnessDay(
                date=d3,
                sleep_hours=8.0,
                wake_hour=8.5,
                steps=8500,
                stress=0.15,
                mental_tiredness=0.20,
                focus=0.80,
                energy=0.85,
                overwhelmed=0.10,
                tasks_due_24h=1,
                minutes_due_24h=45,
                tasks_overdue=0,
                tasks_pending=1,
            ),
            ManualWellnessDay(
                date=d2,
                sleep_hours=8.5,
                wake_hour=8.5,
                steps=9200,
                stress=0.10,
                mental_tiredness=0.15,
                focus=0.85,
                energy=0.90,
                overwhelmed=0.10,
                tasks_due_24h=1,
                minutes_due_24h=30,
                tasks_overdue=0,
                tasks_pending=1,
            ),
            ManualWellnessDay(
                date=d1,
                sleep_hours=8.0,
                wake_hour=7.5,
                steps=7800,
                stress=0.18,
                mental_tiredness=0.25,
                focus=0.75,
                energy=0.80,
                overwhelmed=0.15,
                tasks_due_24h=2,
                minutes_due_24h=60,
                tasks_overdue=0,
                tasks_pending=2,
            ),
        ],
    }
