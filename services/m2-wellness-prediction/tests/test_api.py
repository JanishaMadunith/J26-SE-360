import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.predictor import Predictor
from wellness_contracts import WellnessForecast, WellnessForecastRequest

UTC = UTC
# 08:00 Colombo = 02:30 UTC
AS_OF = datetime(2026, 10, 17, 2, 30, tzinfo=UTC)
fallback_predictor = Predictor()
fallback_predictor.models = None
client = TestClient(create_app(predictor=fallback_predictor))


def body(**kw):
    b = {"user_id": str(uuid.uuid4()), "as_of": AS_OF.isoformat(), "timezone": "Asia/Colombo"}
    b.update(kw)
    return b


def manual_inputs(**current):
    defaults = {
        "date": "2026-10-17",
        "sleep_hours": 7.0,
        "wake_hour": 7.0,
        "steps": 6000,
        "stress": 0.4,
        "tasks_due_24h": 2,
        "minutes_due_24h": 120,
        "tasks_overdue": 0,
        "tasks_pending": 2,
    }
    defaults.update(current)
    return {
        "history": [
            {**defaults, "date": "2026-10-15", "stress": 0.3},
            {**defaults, "date": "2026-10-16", "stress": 0.5},
        ],
        "current": defaults,
    }


def screen_night(sleep_hours: float):
    """Screen off at 23:00 Colombo the previous night, unlock `sleep_hours` later."""
    off = datetime(2026, 10, 16, 17, 30, tzinfo=UTC)  # 23:00 Colombo
    return [
        {"at": (off - timedelta(hours=1)).isoformat(), "type": "unlock"},
        {"at": off.isoformat(), "type": "screen_off"},
        {"at": (off + timedelta(hours=sleep_hours)).isoformat(), "type": "unlock"},
    ]


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_minimal_request_gives_24_hourly_slots():
    r = client.post("/v1/wellness-forecast", json=body())
    assert r.status_code == 200
    fc = WellnessForecast.model_validate(r.json())
    assert len(fc.slots) == 24
    assert fc.sleep_hours_last_night is None
    for a, b in zip(fc.slots, fc.slots[1:], strict=False):
        assert a.end == b.start
    assert all(0 <= s.energy <= 1 and 0 <= s.burnout_risk <= 1 for s in fc.slots)


def test_slot_granularity_and_horizon():
    fc = client.post("/v1/wellness-forecast", json=body(horizon_hours=12, slot_minutes=30)).json()
    assert len(fc["slots"]) == 24 and fc["slot_minutes"] == 30


def test_forecast_has_wellness_output_shape():
    forecast = client.post("/v1/wellness-forecast", json=body()).json()

    assert set(forecast) == {
        "user_id",
        "generated_at",
        "slot_minutes",
        "sleep_hours_last_night",
        "slots",
        "model",
    }
    assert set(forecast["model"]) == {"name", "version"}
    assert set(forecast["slots"][0]) == {
        "start",
        "end",
        "energy",
        "focus",
        "mental_fatigue",
        "burnout_risk",
    }


def test_sleep_estimate_and_sleep_debt_effect():
    good = WellnessForecast.model_validate(
        client.post("/v1/wellness-forecast", json=body(screen_events=screen_night(8))).json()
    )
    poor = WellnessForecast.model_validate(
        client.post("/v1/wellness-forecast", json=body(screen_events=screen_night(4))).json()
    )
    assert abs(good.sleep_hours_last_night - 8) < 0.01
    assert abs(poor.sleep_hours_last_night - 4) < 0.01
    assert sum(s.energy for s in poor.slots) < sum(s.energy for s in good.slots)
    assert sum(s.mental_fatigue for s in poor.slots) > sum(s.mental_fatigue for s in good.slots)


def test_fatigue_builds_through_the_day():
    fc = WellnessForecast.model_validate(client.post("/v1/wellness-forecast", json=body()).json())
    morning = fc.slots[1].mental_fatigue  # ~09:00 local
    late = fc.slots[14].mental_fatigue  # ~22:00 local
    assert late > morning


def test_stress_history_and_overdue_raise_burnout():
    hist = [
        {
            "user_id": str(uuid.uuid4()),
            "window_start": (AS_OF - timedelta(hours=h + 1)).isoformat(),
            "window_end": (AS_OF - timedelta(hours=h)).isoformat(),
            "label": "stress",
            "probabilities": {"stress": 0.8, "relaxed": 0.05, "happy": 0.05, "sad": 0.05, "anxiety": 0.05},
            "confidence": 0.8,
            "model": {"name": "t", "version": "0"},
        }
        for h in range(10)
    ]
    calm = client.post("/v1/wellness-forecast", json=body()).json()
    stressed = client.post(
        "/v1/wellness-forecast",
        json=body(affective_history=hist, workload={"tasks_overdue": 4, "minutes_due_24h": 400}),
    ).json()
    assert stressed["slots"][0]["burnout_risk"] > calm["slots"][0]["burnout_risk"] + 0.3


def test_self_report_pulls_energy():
    rep = [{"at": (AS_OF - timedelta(minutes=10)).isoformat(), "energy": 1}]
    low = client.post("/v1/wellness-forecast", json=body(self_reports=rep)).json()
    base = client.post("/v1/wellness-forecast", json=body()).json()
    assert low["slots"][0]["energy"] < base["slots"][0]["energy"]


def test_trained_predictor_handles_empty_request():
    predictor = Predictor()
    if predictor.models is None:
        pytest.skip("trained model files are not available")

    request = WellnessForecastRequest.model_validate(body())
    forecast = predictor.predict(request)

    assert len(forecast.slots) == 24
    for slot in forecast.slots:
        assert all(
            0 <= value <= 1
            for value in (
                slot.mental_fatigue,
                slot.focus,
                slot.energy,
                slot.burnout_risk,
            )
        )


def test_manual_inputs_change_trained_prediction():
    predictor = Predictor()
    if predictor.models is None:
        pytest.skip("trained model files are not available")

    request = WellnessForecastRequest.model_validate(
        body(manual_inputs=manual_inputs(sleep_hours=4.0, stress=0.9, tasks_overdue=4))
    )
    forecast = predictor.predict(request)

    assert len(forecast.slots) == 24
    assert forecast.sleep_hours_last_night == 4.0
    assert forecast.model.name == "m2-hgb-slot-forecaster"
    assert all(0 <= slot.energy <= 1 for slot in forecast.slots)


def test_trained_predictor_uses_sleep_recovery():
    predictor = Predictor()
    if predictor.models is None:
        pytest.skip("trained model files are not available")

    request = WellnessForecastRequest.model_validate(
        body(
            as_of="2026-10-17T21:00:00+00:00",
            timezone="UTC",
            manual_inputs=manual_inputs(sleep_hours=6.5),
        )
    )
    forecast = predictor.predict(request)
    sleeping = [slot for slot in forecast.slots if slot.start.hour >= 23 or slot.start.hour < 7]
    waking = [slot for slot in forecast.slots if slot.start.hour == 7]

    assert sleeping
    assert waking
    assert all(slot.energy <= 0.2 and slot.focus <= 0.05 for slot in sleeping)
    assert sleeping[-1].mental_fatigue <= sleeping[0].mental_fatigue
    assert waking[0].energy >= 0.45
