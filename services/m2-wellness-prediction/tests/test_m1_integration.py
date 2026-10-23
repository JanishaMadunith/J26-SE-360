"""Tests verifying M1 affective state to M2 cognitive wellness integration."""

import uuid
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from app.predictor import Predictor
from wellness_contracts import (
    ManualWellnessInputs,
    WellnessForecast,
    WellnessForecastRequest,
    create_current_manual_wellness_day,
    get_preset_history_scenarios,
    m1_fused_to_affective_state_result,
    map_m1_to_current_wellness,
)


def test_map_m1_to_current_wellness_classes():
    # Angry/Stressed dominant
    angry_probs = [0.05, 0.05, 0.05, 0.15, 0.70]
    angry_derived = map_m1_to_current_wellness(angry_probs, {"jitter": 60, "error_rate": 15})
    assert angry_derived["stress"] > 0.65
    assert angry_derived["overwhelmed"] > 0.70

    # Calm dominant
    calm_probs = [0.10, 0.70, 0.15, 0.03, 0.02]
    calm_derived = map_m1_to_current_wellness(calm_probs, {"jitter": 25, "error_rate": 2})
    assert calm_derived["focus"] > 0.60
    assert calm_derived["stress"] < 0.15

    # Sad dominant
    sad_probs = [0.05, 0.05, 0.05, 0.75, 0.10]
    sad_derived = map_m1_to_current_wellness(sad_probs, {"hold_time": 140, "wpm": 25})
    assert sad_derived["mental_tiredness"] > 0.60
    assert sad_derived["energy"] < 0.40


def test_preset_history_scenarios():
    presets = get_preset_history_scenarios(date.today())
    assert "Balanced Normal Week" in presets
    assert "Exam / Crunch Week (High Stress)" in presets
    assert "Post-Weekend Recovery (Low Stress)" in presets

    for name, history in presets.items():
        assert len(history) == 3
        # Dates should be descending / past 3 days
        assert history[0].date < history[1].date < history[2].date < date.today()
        for day in history:
            assert 0 <= day.stress <= 1
            assert 0 <= day.mental_tiredness <= 1
            assert 0 <= day.energy <= 1
            assert 0 <= day.focus <= 1


def test_m1_fused_to_affective_state_result():
    now = datetime.now(UTC)
    result = m1_fused_to_affective_state_result(
        user_id=uuid.uuid4(),
        window_start=now - timedelta(minutes=60),
        window_end=now,
        fused_emotion="Angry",
        fused_confidence=0.82,
        fused_probs=[0.05, 0.05, 0.05, 0.15, 0.70],
        features={"wpm": 38.0, "error_rate": 12.0},
    )
    assert result.label.value == "stress"
    assert result.confidence == 0.82
    assert "stress" in result.probabilities


def test_end_to_end_m1_current_with_past_history_forecast():
    predictor = Predictor()
    today = date(2026, 10, 23)
    presets = get_preset_history_scenarios(today)
    past_history = presets["Balanced Normal Week"]

    # Current day seeded with M1 Calm output
    calm_current = create_current_manual_wellness_day(
        target_date=today,
        fused_probs=[0.10, 0.70, 0.15, 0.03, 0.02],
        typing_metrics={"wpm": 45, "hold_time": 80, "jitter": 25, "error_rate": 2},
        sleep_hours=7.5,
    )

    as_of = datetime(2026, 10, 23, 10, 0, tzinfo=ZoneInfo("Asia/Colombo"))
    calm_req = WellnessForecastRequest(
        user_id=uuid.uuid4(),
        as_of=as_of,
        timezone="Asia/Colombo",
        horizon_hours=24,
        slot_minutes=60,
        manual_inputs=ManualWellnessInputs(history=past_history, current=calm_current),
    )
    calm_forecast = predictor.predict(calm_req)

    # Current day seeded with M1 Stressed output
    stressed_current = create_current_manual_wellness_day(
        target_date=today,
        fused_probs=[0.05, 0.05, 0.05, 0.15, 0.70],
        typing_metrics={"wpm": 38, "hold_time": 90, "jitter": 65, "error_rate": 15},
        sleep_hours=5.0,
    )
    stressed_req = WellnessForecastRequest(
        user_id=uuid.uuid4(),
        as_of=as_of,
        timezone="Asia/Colombo",
        horizon_hours=24,
        slot_minutes=60,
        manual_inputs=ManualWellnessInputs(history=past_history, current=stressed_current),
    )
    stressed_forecast = predictor.predict(stressed_req)

    assert len(calm_forecast.slots) == 24
    assert len(stressed_forecast.slots) == 24

    # Stressed current day should show higher mental fatigue and burnout risk than calm day
    assert stressed_forecast.slots[0].mental_fatigue > calm_forecast.slots[0].mental_fatigue
    assert stressed_forecast.slots[0].burnout_risk > calm_forecast.slots[0].burnout_risk
    assert calm_forecast.slots[0].focus > stressed_forecast.slots[0].focus
