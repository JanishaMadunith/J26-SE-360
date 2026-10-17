import math
import uuid
from datetime import UTC, datetime, timedelta

import numpy as np
from fastapi.testclient import TestClient

from app.main import create_app
from wellness_contracts import AffectiveLabel, AffectiveStateResult

NOW = datetime(2026, 10, 17, 10, 0, tzinfo=UTC)
client = TestClient(create_app())


def ppg(bpm: float, seconds: float = 60, fps: float = 30.0, jitter: float = 0.03, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 1 / fps)
    # beat-to-beat variability via a slowly varying instantaneous frequency
    f = bpm / 60 * (1 + jitter * np.sin(2 * math.pi * 0.25 * t))
    phase = 2 * math.pi * np.cumsum(f) / fps
    red = 180 + 2.0 * np.sin(phase) + 0.3 * rng.standard_normal(len(t)) + 0.02 * t
    green = 60 + 1.5 * np.sin(phase) + 0.3 * rng.standard_normal(len(t))
    return {
        "started_at": (NOW - timedelta(minutes=20)).isoformat(),
        "fps": fps,
        "t_ms": [int(x * 1000) for x in t],
        "red": red.round(3).tolist(),
        "green": green.round(3).tolist(),
        "blue": [20.0] * len(t),
    }


def body(**kw):
    b = {
        "user_id": str(uuid.uuid4()),
        "window_start": (NOW - timedelta(hours=1)).isoformat(),
        "window_end": NOW.isoformat(),
    }
    b.update(kw)
    return b


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert r.json()["service"] == "m1-affective-state"


def test_empty_signals_returns_valid_result():
    r = client.post("/v1/affective-state", json=body())
    assert r.status_code == 200
    res = AffectiveStateResult.model_validate(r.json())
    assert abs(sum(res.probabilities.values()) - 1) < 1e-3
    assert res.features["hr_bpm"] is None


def test_heart_rate_extracted_from_raw_ppg():
    for bpm in (60, 72, 95):
        r = client.post("/v1/affective-state", json=body(ppg=[ppg(bpm)]))
        assert r.status_code == 200
        f = r.json()["features"]
        assert abs(f["hr_bpm"] - bpm) < 4, (bpm, f["hr_bpm"])
        assert f["hrv_rmssd_ms"] is None or 0 < f["hrv_rmssd_ms"] < 200


def test_short_ppg_is_ignored():
    r = client.post("/v1/affective-state", json=body(ppg=[ppg(72, seconds=5)]))
    assert r.json()["features"]["hr_bpm"] is None


def test_typing_and_phone_use_features():
    keys = []
    t = NOW - timedelta(minutes=30)
    for i in range(200):
        t += timedelta(milliseconds=180 + (i % 7) * 20)
        keys.append({"down_at": t.isoformat(), "hold_ms": 90, "key_class": "backspace" if i % 10 == 0 else "letter"})
    screen = [{"at": (NOW - timedelta(minutes=50 - i * 5)).isoformat(), "type": "unlock"} for i in range(8)]
    r = client.post("/v1/affective-state", json=body(keystrokes=keys, screen_events=screen))
    f = r.json()["features"]
    assert 200 < f["typing_speed_kpm"] < 350
    assert abs(f["backspace_rate"] - 0.1) < 0.01
    assert abs(f["unlocks_per_hour"] - 8) < 0.01


def test_stress_pattern_ranks_stress_high():
    keys = [
        {
            "down_at": (NOW - timedelta(seconds=300 - i)).isoformat(),
            "hold_ms": 80,
            "key_class": "backspace" if i % 4 == 0 else "letter",
        }
        for i in range(120)
    ]
    unlocks = [{"at": (NOW - timedelta(minutes=58 - i * 4)).isoformat(), "type": "unlock"} for i in range(14)]
    r = client.post(
        "/v1/affective-state", json=body(ppg=[ppg(105, jitter=0.005)], keystrokes=keys, screen_events=unlocks)
    )
    res = AffectiveStateResult.model_validate(r.json())
    top2 = sorted(res.probabilities, key=res.probabilities.get, reverse=True)[:2]
    assert AffectiveLabel.stress in top2 or AffectiveLabel.anxiety in top2


def test_rejects_unknown_fields():
    r = client.post("/v1/affective-state", json=body(typed_text="secret"))
    assert r.status_code == 422
