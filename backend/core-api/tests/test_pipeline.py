import json
import uuid
from datetime import timedelta

from sqlalchemy import select

from app.models import Task, TaskEvent

from conftest import iso, now_utc, sample_batch


def _task(client, h, **kw):
    body = {"title": "Task", "category": "study", "estimated_minutes": 60, "priority": 3, **kw}
    return client.post("/v1/tasks", json=body, headers=h).json()


def _upload_and_settle(client, h):
    """Upload one batch; the first upload auto-runs the pipeline in the background."""
    client.post("/v1/sensing/batches", json=sample_batch(), headers=h)


def test_full_pipeline_happy_path(client, user, db, fake_modules):
    h = user["headers"]
    t1 = _task(client, h, title="Lab report", deadline=iso(now_utc() + timedelta(days=2)))
    t2 = _task(
        client,
        h,
        title="Lecture",
        fixed_start=iso(now_utc() + timedelta(hours=5)),
        fixed_end=iso(now_utc() + timedelta(hours=7)),
    )
    client.post("/v1/sensing/batches", json=sample_batch(), headers=h)
    fake_modules.calls.clear()

    run = client.post("/v1/pipeline/run", headers=h).json()
    assert run["status"] == "ok" and run["trigger"] == "manual"
    assert [s["step"] for s in run["steps"]] == ["affective", "wellness", "schedule", "recommendations"]
    assert all(s["ok"] and not s["used_fallback"] for s in run["steps"])
    assert fake_modules.calls == ["m1", "m2", "m3", "m4"]

    sched = client.get("/v1/schedule/current", headers=h).json()
    assert {i["task_id"] for i in sched["items"]} == {t1["id"], t2["id"]}
    statuses = {t["id"]: t["status"] for t in client.get("/v1/tasks", headers=h).json()}
    assert statuses == {t1["id"]: "scheduled", t2["id"]: "scheduled"}

    ins = client.get("/v1/insights/latest", headers=h).json()
    assert ins["affective"]["label"] == "relaxed" and len(ins["wellness"]["slots"]) == 24
    rec = client.get("/v1/recommendations/latest", headers=h).json()
    assert rec["meals"][0]["meal_type"] == "lunch"

    runs = client.get("/v1/pipeline/runs", headers=h).json()
    assert len(runs) == 2 and runs[0]["trigger"] == "manual" and runs[1]["trigger"] == "sensor_upload"


def test_module_requests_carry_the_right_data(client, user, fake_modules):
    h = user["headers"]
    _task(client, h, deadline=iso(now_utc() + timedelta(hours=10)), estimated_minutes=45)
    client.post("/v1/sensing/batches", json=sample_batch(), headers=h)
    m1 = json.loads(fake_modules.last_requests["m1"])
    assert len(m1["keystrokes"]) == 5 and len(m1["ppg"]) == 1 and len(m1["screen_events"]) == 2
    assert "character" not in m1["keystrokes"][0]
    m2 = json.loads(fake_modules.last_requests["m2"])
    assert m2["workload"] == {"tasks_due_24h": 1, "minutes_due_24h": 45, "tasks_overdue": 0, "tasks_pending": 1}
    assert len(m2["affective_history"]) == 1 and m2["slot_minutes"] == 60
    m4 = json.loads(fake_modules.last_requests["m4"])
    assert m4["schedule"] is not None and m4["affective_state"]["label"] == "relaxed"


def test_wellness_down_uses_fallback(client, user, fake_modules):
    h = user["headers"]
    _task(client, h)
    _upload_and_settle(client, h)  # first run: everything up, results stored
    fake_modules.down.add("m2")
    run = client.post("/v1/pipeline/run", headers=h).json()
    assert run["status"] == "partial"
    steps = {s["step"]: s for s in run["steps"]}
    assert steps["wellness"]["ok"] is False and steps["wellness"]["used_fallback"] is True
    assert "m2" in steps["wellness"]["error"]
    assert steps["schedule"]["ok"] and steps["recommendations"]["ok"]  # the rest still ran


def test_all_modules_down_without_history(client, user, fake_modules):
    h = user["headers"]
    fake_modules.down.update({"m1", "m2", "m3", "m4"})
    client.post("/v1/sensing/batches", json=sample_batch(), headers=h)
    run = client.post("/v1/pipeline/run", headers=h).json()
    assert run["status"] == "failed"
    assert all(not s["ok"] and not s["used_fallback"] for s in run["steps"])
    assert client.get("/v1/schedule/current", headers=h).status_code == 404


def test_no_sensor_data_skips_m1(client, user, fake_modules):
    h = user["headers"]
    run = client.post("/v1/pipeline/run", headers=h).json()
    aff = run["steps"][0]
    assert aff["ok"] is True and aff["error"].startswith("skipped")
    assert "m1" not in fake_modules.calls and fake_modules.calls == ["m2", "m3", "m4"]


def test_overdue_tasks_become_missed(client, user, db):
    h = user["headers"]
    t = _task(client, h, title="overdue", deadline=iso(now_utc() - timedelta(hours=1)))
    client.post("/v1/pipeline/run", headers=h)
    assert {x["id"]: x["status"] for x in client.get("/v1/tasks", headers=h).json()}[t["id"]] == "missed"
    events = [e.event for e in db.scalars(select(TaskEvent).where(TaskEvent.task_id == uuid.UUID(t["id"])))]
    assert events == ["created", "missed"]


def test_rescheduling_is_logged(client, user, db):
    h = user["headers"]
    t = _task(client, h, deadline=iso(now_utc() + timedelta(days=3)))
    client.post("/v1/pipeline/run", headers=h)
    client.post("/v1/pipeline/run", headers=h)  # fake M3 places tasks relative to 'now' -> slot moves
    events = [
        e.event
        for e in db.scalars(select(TaskEvent).where(TaskEvent.task_id == uuid.UUID(t["id"])).order_by(TaskEvent.at))
    ]
    assert events == ["created", "scheduled", "rescheduled"]
    db.expire_all()
    assert db.get(Task, uuid.UUID(t["id"])).times_rescheduled == 1
