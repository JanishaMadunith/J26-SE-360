import uuid
from datetime import timedelta

from sqlalchemy import select

from app.models import TaskEvent

from conftest import auth_headers, iso, now_utc, register


def _create(client, h, **kw):
    body = {"title": "Write lit review", "category": "study", "estimated_minutes": 90, "priority": 4, **kw}
    r = client.post("/v1/tasks", json=body, headers=h)
    assert r.status_code == 201, r.text
    return r.json()


def _events(db, task_id):
    return [
        e.event
        for e in db.scalars(select(TaskEvent).where(TaskEvent.task_id == uuid.UUID(task_id)).order_by(TaskEvent.at))
    ]


def test_create_list_order(client, user):
    h = user["headers"]
    later = _create(client, h, title="later", deadline=iso(now_utc() + timedelta(days=3)))
    _create(client, h, title="sooner", deadline=iso(now_utc() + timedelta(days=1)))
    nodl = _create(client, h, title="no deadline")
    assert later["status"] == "pending"
    titles = [t["title"] for t in client.get("/v1/tasks", headers=h).json()]
    assert titles == ["sooner", "later", "no deadline"]
    assert nodl["deadline"] is None


def test_validation(client, user):
    h = user["headers"]
    assert client.post("/v1/tasks", json={"title": "", "estimated_minutes": 30}, headers=h).status_code == 422
    assert client.post("/v1/tasks", json={"title": "x", "estimated_minutes": 2}, headers=h).status_code == 422
    half_fixed = {"title": "lecture", "estimated_minutes": 60, "fixed_start": iso(now_utc())}
    assert client.post("/v1/tasks", json=half_fixed, headers=h).status_code == 422


def test_update_complete_delete_and_events(client, user, db):
    h = user["headers"]
    t = _create(client, h)
    r = client.patch(f"/v1/tasks/{t['id']}", json={"title": "Write literature review", "priority": 5}, headers=h)
    assert r.status_code == 200 and r.json()["title"] == "Write literature review" and r.json()["priority"] == 5

    done = client.post(f"/v1/tasks/{t['id']}/complete", headers=h).json()
    assert done["status"] == "done" and done["completed_at"] is not None
    assert _events(db, done["id"]) == ["created", "completed"]

    t2 = _create(client, h, title="to cancel")
    assert client.delete(f"/v1/tasks/{t2['id']}", headers=h).status_code == 204
    cancelled = client.get("/v1/tasks", params={"status": "cancelled"}, headers=h).json()
    assert [c["id"] for c in cancelled] == [t2["id"]]
    assert _events(db, t2["id"]) == ["created", "cancelled"]

    t3 = _create(client, h, title="status via patch")
    client.patch(f"/v1/tasks/{t3['id']}", json={"status": "done"}, headers=h)
    assert _events(db, t3["id"]) == ["created", "completed"]


def test_other_users_tasks_are_invisible(client, user):
    t = _create(client, user["headers"])
    other = auth_headers(register(client))
    assert client.patch(f"/v1/tasks/{t['id']}", json={"title": "hijack"}, headers=other).status_code == 404
    assert client.post(f"/v1/tasks/{t['id']}/complete", headers=other).status_code == 404
    assert client.get("/v1/tasks", headers=other).json() == []
