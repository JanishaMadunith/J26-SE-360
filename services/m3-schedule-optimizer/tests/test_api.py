from fastapi.testclient import TestClient

from app.main import create_app
from wellness_contracts import ScheduleResult

client = TestClient(create_app())


def test_health():
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["service"] == "m3-schedule-optimizer"
    assert r["model"]["name"] == "m3-greedy-energy-aware"


def test_schedule_endpoint_returns_contract(helpers):
    ts = [
        helpers.task("read papers", minutes=90, priority=5, deadline=helpers.local(1, 18)),
        helpers.task("gym", category="exercise", minutes=60),
        helpers.task("lecture", category="study", fixed=(helpers.local(0, 10), helpers.local(0, 12))),
    ]
    r = client.post(
        "/v1/schedule",
        json=helpers.request(ts, wellness=lambda uid: helpers.forecast(uid, lambda h: (0.6, 0.6, 0.4, 0.3))),
    )
    assert r.status_code == 200, r.text
    res = ScheduleResult.model_validate(r.json())
    assert len(res.items) == 3 and sum(i.is_fixed for i in res.items) == 1
    assert all(0 <= i.completion_probability <= 1 for i in res.items)


def test_schedule_without_wellness_or_tasks(helpers):
    r = client.post("/v1/schedule", json=helpers.request([]))
    assert r.status_code == 200 and r.json()["items"] == []


def test_invalid_request_rejected(helpers):
    req = helpers.request([helpers.task(minutes=1)])  # below the 5-minute minimum
    assert client.post("/v1/schedule", json=req).status_code == 422
