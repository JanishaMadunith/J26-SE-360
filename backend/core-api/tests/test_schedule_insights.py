from datetime import timedelta

from conftest import iso, now_utc


def test_schedule_404_then_regenerate(client, user, fake_modules):
    h = user["headers"]
    assert client.get("/v1/schedule/current", headers=h).status_code == 404
    client.post(
        "/v1/tasks",
        json={
            "title": "Gym",
            "category": "exercise",
            "estimated_minutes": 45,
            "deadline": iso(now_utc() + timedelta(days=1)),
        },
        headers=h,
    )
    r = client.post("/v1/schedule/regenerate", headers=h)
    assert r.status_code == 200 and r.json()["reason"] == "manual" and len(r.json()["items"]) == 1
    assert fake_modules.calls == ["m3"]  # regenerate runs Module 3 only
    current = client.get("/v1/schedule/current", headers=h).json()
    assert current["items"][0]["task_id"] == r.json()["items"][0]["task_id"]


def test_regenerate_when_scheduler_down(client, user, fake_modules):
    fake_modules.down.add("m3")
    r = client.post("/v1/schedule/regenerate", headers=user["headers"])
    assert r.status_code == 503 and "m3" in r.json()["detail"]


def test_insights_and_recommendations_empty(client, user):
    h = user["headers"]
    assert client.get("/v1/insights/latest", headers=h).json() == {"affective": None, "wellness": None}
    assert client.get("/v1/recommendations/latest", headers=h).status_code == 404


def test_health(client, fake_modules):
    body = client.get("/health").json()
    assert body == {"status": "ok", "db": True, "modules": {"m1": True, "m2": True, "m3": True, "m4": True}}
    fake_modules.down.add("m4")
    body = client.get("/health").json()
    assert body["status"] == "degraded" and body["modules"]["m4"] is False


def test_pipeline_runs_limit(client, user):
    h = user["headers"]
    for _ in range(3):
        client.post("/v1/pipeline/run", headers=h)
    assert len(client.get("/v1/pipeline/runs", params={"limit": 2}, headers=h).json()) == 2
    assert client.get("/v1/pipeline/runs", params={"limit": 0}, headers=h).status_code == 422
