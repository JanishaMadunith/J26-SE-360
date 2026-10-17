import uuid

from sqlalchemy import func, select

from app.models import PipelineRun, PPGRecording, SelfReport, SensorEvent

from conftest import auth_headers, now_utc, register, sample_batch


def test_upload_batch_counts_and_storage(client, user, db):
    batch = sample_batch()
    r = client.post("/v1/sensing/batches", json=batch, headers=user["headers"])
    assert r.status_code == 202, r.text
    ack = r.json()
    assert ack["accepted"] is True and ack["duplicate"] is False
    assert ack["counts"] == {
        "keystroke": 5,
        "screen": 2,
        "app_usage": 1,
        "activity": 1,
        "steps": 1,
        "light": 1,
        "ppg": 1,
        "self_reports": 1,
    }
    assert db.scalar(select(func.count()).select_from(SensorEvent)) == 11
    assert db.scalar(select(func.count()).select_from(PPGRecording)) == 1
    assert db.scalar(select(func.count()).select_from(SelfReport)) == 1
    kinds = {k for (k,) in db.execute(select(SensorEvent.kind).distinct())}
    assert kinds == {"keystroke", "screen", "app_usage", "activity", "steps", "light"}


def test_duplicate_batch_is_idempotent(client, user, db):
    batch = sample_batch()
    first = client.post("/v1/sensing/batches", json=batch, headers=user["headers"]).json()
    second = client.post("/v1/sensing/batches", json=batch, headers=user["headers"])
    assert second.status_code == 202
    assert second.json()["duplicate"] is True and second.json()["counts"] == first["counts"]
    assert db.scalar(select(func.count()).select_from(SensorEvent)) == 11  # nothing stored twice


def test_batch_id_of_another_user_conflicts(client, user):
    batch = sample_batch()
    client.post("/v1/sensing/batches", json=batch, headers=user["headers"])
    other = auth_headers(register(client))
    assert client.post("/v1/sensing/batches", json=batch, headers=other).status_code == 409


def test_typed_text_is_rejected(client, user):
    """Privacy guard: the contract forbids unknown fields, so typed content can never be uploaded."""
    batch = sample_batch()
    batch["keystrokes"][0]["character"] = "a"
    assert client.post("/v1/sensing/batches", json=batch, headers=user["headers"]).status_code == 422
    batch = sample_batch()
    batch["typed_text"] = "my password is hunter2"
    assert client.post("/v1/sensing/batches", json=batch, headers=user["headers"]).status_code == 422


def test_invalid_ppg_rejected(client, user):
    batch = sample_batch()
    batch["ppg"][0]["red"] = [1.0]  # length mismatch
    assert client.post("/v1/sensing/batches", json=batch, headers=user["headers"]).status_code == 422


def test_first_upload_triggers_pipeline_once(client, user, db, fake_modules):
    client.post("/v1/sensing/batches", json=sample_batch(), headers=user["headers"])
    runs = db.scalars(select(PipelineRun)).all()
    assert len(runs) == 1 and runs[0].trigger == "sensor_upload" and runs[0].status == "ok"
    assert fake_modules.calls == ["m1", "m2", "m3", "m4"]
    # a second upload within PIPELINE_MIN_INTERVAL_MINUTES does not start another run
    client.post("/v1/sensing/batches", json=sample_batch(), headers=user["headers"])
    db.expire_all()
    assert db.scalar(select(func.count()).select_from(PipelineRun)) == 1


def test_self_report(client, user):
    body = {"at": now_utc().isoformat(), "mood": 3, "stress": 4}
    r = client.post("/v1/self-reports", json=body, headers=user["headers"])
    assert r.status_code == 201 and r.json()["stress"] == 4
    assert client.post("/v1/self-reports", json={**body, "mood": 9}, headers=user["headers"]).status_code == 422


def test_uuid_batch_id_required(client, user):
    batch = sample_batch(batch_id=str(uuid.uuid4()))
    batch["batch_id"] = "not-a-uuid"
    assert client.post("/v1/sensing/batches", json=batch, headers=user["headers"]).status_code == 422
