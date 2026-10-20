"""Tests for the trained-model layer. Model-dependent tests are skipped when models/ has no trained files
(e.g. in CI or a fresh clone); the scheduler must then still work with the baseline."""

import math

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.predictor import Predictor
from m3.completion_model import BehaviourProfile, features_from_app
from m3.context_datasets import CIRCADIAN, forecast_from_days, hourly_wellness
from m3.scheduling.heuristic import COEF, CompletionFeatures, TaskPrior, completion_probability, schedule
from m3.tasktime import PARTS, part_of_hour, votes
from wellness_contracts import ScheduleRequest, TaskCategory, WellnessForecast

PRED = Predictor()
client = TestClient(create_app(PRED))
needs_completion = pytest.mark.skipif(not PRED.has_completion, reason="trained completion model not in models/")
needs_tasktime = pytest.mark.skipif(not PRED.has_tasktime, reason="task time-of-day model not in models/")


def _f(**kw):
    base = {
        "slack_hours": 24, "energy": 0.6, "focus": 0.6, "fatigue": 0.4, "duration_min": 60, "priority": 3,
        "category": TaskCategory.study, "category_rate": 0.7, "times_rescheduled": 0, "local_start_hour": 10,
    }  # fmt: skip
    base.update(kw)
    return CompletionFeatures(**base)


def test_prior_logit_is_used_and_slot_adjusts_it():
    p_hi = completion_probability(_f(base_logit=math.log(0.9 / 0.1), slack_now_hours=24))
    p_lo = completion_probability(_f(base_logit=math.log(0.2 / 0.8), slack_now_hours=24))
    assert p_hi > 0.8 > 0.3 > p_lo
    tired = completion_probability(_f(base_logit=0.0, slack_now_hours=24, energy=0.2, focus=0.2, fatigue=0.8))
    fresh = completion_probability(_f(base_logit=0.0, slack_now_hours=24, energy=0.8, focus=0.8, fatigue=0.2))
    assert fresh > 0.5 > tired


def test_time_of_day_preference_raises_chance():
    good = completion_probability(_f(time_pref=0.8))
    bad = completion_probability(_f(time_pref=0.05))
    assert good > bad and COEF["time_fit"] > 0


def test_task_time_votes_and_hours():
    t = {"TimeJudgements": [{"Known": "yes", "Times": "WD-morning"}, {"Known": "yes", "Times": "WE-anytime"},
                            {"Known": "no", "Times": "WD-night"}]}  # fmt: skip
    v = votes(t)
    assert v[PARTS.index("morning")] == pytest.approx(1.25) and v.sum() == pytest.approx(2.0)
    assert part_of_hour(9) == "morning" and part_of_hour(14) == "afternoon" and part_of_hour(22) == "night"


def test_wellness_standin_matches_module2_contract():
    day = {"energy": 0.7, "focus": 0.6, "calm": 0.5, "sleep_duration_h": 7.0}
    assert len(CIRCADIAN) == 24
    assert hourly_wellness(day, 11)["energy"] > hourly_wellness(day, 3)["energy"]
    from datetime import datetime
    from zoneinfo import ZoneInfo

    fc = forecast_from_days(
        [day, day],
        "00000000-0000-0000-0000-000000000001",
        datetime(2026, 10, 20, 7, tzinfo=ZoneInfo("Asia/Colombo")),
        30,
    )
    WellnessForecast.model_validate(fc)  # valid Module 2 format


def test_features_from_app_shape():
    f = features_from_app(BehaviourProfile(on_time_rate=0.9, days_since_active=1), 3, 480, 5)
    assert f["H"] == 3 and f["is_tma"] == 1 and f["weight"] == 25.0 and f["prior_on_time_rate"] == 0.9
    assert features_from_app(BehaviourProfile(), 1, 60, 3)["H"] == 8  # small task: 1 day is plenty of time
    assert features_from_app(BehaviourProfile(), 10, 60, 3)["H"] == 14  # capped at OULAD's range
    new = features_from_app(BehaviourProfile(), 3, 60, 3)
    assert new["prior_on_time_rate"] is None and new["n_prior"] == 0  # no history -> "unknown", as in training


def test_scheduler_accepts_priors(helpers):
    t1 = helpers.task("write report", minutes=60, deadline=helpers.local(1, 18))
    req = ScheduleRequest.model_validate(helpers.request([t1]))
    tid = req.tasks[0].task_id
    base = schedule(req)
    ml = schedule(
        req, {tid: TaskPrior(p_on_time=0.95, parts={"morning": 0.9, "afternoon": 0.05, "evening": 0.03, "night": 0.02})}
    )
    assert len(base.items) == len(ml.items) == 1
    assert ml.items[0].completion_probability > 0.8
    assert ml.items[0].start.astimezone(helpers.AS_OF.tzinfo).hour < 12  # placed in the morning


def test_health_reports_models():
    m = client.get("/health").json()["model"]
    assert m["name"] in ("m3-ml-energy-aware", "m3-greedy-energy-aware") and "completion_model" in m


@needs_completion
def test_completion_endpoint_orders_users_sensibly():
    good = {"on_time_rate": 0.95, "days_since_active": 1, "active_days_7": 6}
    bad = {
        "on_time_rate": 0.3,
        "days_since_active": 10,
        "active_days_7": 1,
        "unfinished_earlier_tasks": 3,
        "overdue_tasks": 2,
    }
    pg = client.post("/v1/completion/predict", json={"profile": good, "task": {}}).json()
    pb = client.post("/v1/completion/predict", json={"profile": bad, "task": {}}).json()
    assert 0 <= pb["probability_on_time"] < pg["probability_on_time"] <= 1
    assert pg["reasons"] and all(r["feature"] != "is_tma" for r in pg["reasons"])


@needs_completion
def test_examples_and_plan_end_to_end():
    ex = client.get("/v1/completion/examples")
    if ex.status_code == 503:
        pytest.skip("demo assets not built")
    r = client.post("/v1/completion/predict", json={"features": ex.json()[0]["features"]})
    assert r.status_code == 200
    plan = client.post(
        "/v1/demo/plan",
        json={
            "tasks": [
                {"title": "write report", "due_in_days": 2},
                {"title": "go to the gym", "category": "exercise", "estimated_minutes": 60},
            ]
        },
    )
    assert plan.status_code == 200, plan.text
    j = plan.json()
    assert len(j["tasks"]) == 2 and j["wellness"]["hourly"]


@needs_tasktime
def test_tasktime_endpoint():
    r = client.post("/v1/tasktime", json=["morning run", "watch a movie"]).json()
    assert r[0]["best"] == "morning" and set(r[1]["parts"]) == set(PARTS)


def test_demo_page_served():
    r = client.get("/demo")
    assert r.status_code == 200 and "Module 3" in r.text
