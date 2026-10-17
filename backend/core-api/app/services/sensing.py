"""Storing phone uploads and rebuilding signal windows for the module services."""

import uuid
from datetime import datetime

from sqlalchemy import insert, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from wellness_contracts import (
    ActivityTransition,
    AppUsageSession,
    KeystrokeEvent,
    LightReading,
    ScreenEvent,
    SensorBatchAck,
    StepCount,
)
from wellness_contracts import PPGRecording as PPGContract
from wellness_contracts import SelfReport as SelfReportContract
from wellness_contracts import SensorBatch as SensorBatchContract

from ..db import utcnow
from ..models import Device, PPGRecording, SelfReport, SensorBatch, SensorEvent, User

# kind -> (contract field on SensorBatch, contract model, timestamp attribute)
EVENT_KINDS = {
    "keystroke": ("keystrokes", KeystrokeEvent, "down_at"),
    "screen": ("screen_events", ScreenEvent, "at"),
    "app_usage": ("app_usage", AppUsageSession, "start"),
    "activity": ("activity", ActivityTransition, "at"),
    "steps": ("steps", StepCount, "start"),
    "light": ("light", LightReading, "at"),
}


class BatchOwnershipError(Exception):
    """The batch_id already exists for a different user."""


def _counts(batch: SensorBatchContract) -> dict[str, int]:
    c = {kind: len(getattr(batch, field)) for kind, (field, _, _) in EVENT_KINDS.items()}
    c["ppg"] = len(batch.ppg)
    c["self_reports"] = len(batch.self_reports)
    return c


def _upsert_device(db: Session, user: User, batch: SensorBatchContract) -> None:
    dev = db.scalar(select(Device).where(Device.user_id == user.id, Device.device_id == batch.device_id))
    if dev is None:
        db.add(Device(user_id=user.id, device_id=batch.device_id, app_version=batch.app_version))
    else:
        dev.app_version = batch.app_version
        dev.last_seen = utcnow()


def _store_self_report(
    db: Session, user_id: uuid.UUID, r: SelfReportContract, batch_id: uuid.UUID | None
) -> SelfReport:
    row = SelfReport(
        user_id=user_id,
        batch_id=batch_id,
        at=r.at,
        mood=r.mood,
        stress=r.stress,
        energy=r.energy,
        focus=r.focus,
        note=r.note,
    )
    db.add(row)
    return row


def ingest_batch(db: Session, user: User, batch: SensorBatchContract) -> SensorBatchAck:
    existing = db.get(SensorBatch, batch.batch_id)
    if existing is not None:
        if existing.user_id != user.id:
            raise BatchOwnershipError(str(batch.batch_id))
        return SensorBatchAck(batch_id=batch.batch_id, accepted=True, duplicate=True, counts=existing.counts)

    counts = _counts(batch)
    db.add(
        SensorBatch(
            id=batch.batch_id,
            user_id=user.id,
            device_id=batch.device_id,
            app_version=batch.app_version,
            timezone=batch.timezone,
            collected_from=batch.collected_from,
            collected_to=batch.collected_to,
            counts=counts,
            payload=batch.model_dump(mode="json"),
        )
    )
    _upsert_device(db, user, batch)
    try:
        db.flush()  # surfaces a concurrent duplicate upload as IntegrityError
    except IntegrityError:
        db.rollback()
        existing = db.get(SensorBatch, batch.batch_id)
        if existing is None or existing.user_id != user.id:
            raise BatchOwnershipError(str(batch.batch_id)) from None
        return SensorBatchAck(batch_id=batch.batch_id, accepted=True, duplicate=True, counts=existing.counts)

    rows = []
    for kind, (field, _, ts_attr) in EVENT_KINDS.items():
        for item in getattr(batch, field):
            value = item.model_dump(mode="json")
            ts = getattr(item, ts_attr)
            value.pop(ts_attr, None)
            rows.append(
                {
                    "id": uuid.uuid4(),
                    "ts": ts,
                    "user_id": user.id,
                    "batch_id": batch.batch_id,
                    "kind": kind,
                    "value": value,
                }
            )
    if rows:
        db.execute(insert(SensorEvent), rows)
    for rec in batch.ppg:
        db.add(
            PPGRecording(
                user_id=user.id,
                batch_id=batch.batch_id,
                started_at=rec.started_at,
                fps=rec.fps,
                n_frames=len(rec.t_ms),
                data={"t_ms": rec.t_ms, "red": rec.red, "green": rec.green, "blue": rec.blue, "torch_on": rec.torch_on},
            )
        )
    for r in batch.self_reports:
        _store_self_report(db, user.id, r, batch.batch_id)
    db.commit()
    return SensorBatchAck(batch_id=batch.batch_id, accepted=True, duplicate=False, counts=counts)


def store_self_report(db: Session, user: User, report: SelfReportContract) -> SelfReportContract:
    _store_self_report(db, user.id, report, None)
    db.commit()
    return report


# ---------------------------------------------------------------- windows for module requests
def load_events(
    db: Session, user_id: uuid.UUID, start: datetime, end: datetime, kinds: tuple[str, ...] = tuple(EVENT_KINDS)
) -> dict[str, list]:
    """Rebuild contract objects for sensor events with start <= ts <= end, keyed by SensorBatch field name."""
    out: dict[str, list] = {EVENT_KINDS[k][0]: [] for k in kinds}
    q = (
        select(SensorEvent.kind, SensorEvent.ts, SensorEvent.value)
        .where(
            SensorEvent.user_id == user_id, SensorEvent.kind.in_(kinds), SensorEvent.ts >= start, SensorEvent.ts <= end
        )
        .order_by(SensorEvent.ts)
    )
    for kind, ts, value in db.execute(q):
        field, model, ts_attr = EVENT_KINDS[kind]
        out[field].append(model.model_validate({**value, ts_attr: ts}))
    return out


def load_ppg(db: Session, user_id: uuid.UUID, start: datetime, end: datetime) -> list[PPGContract]:
    q = (
        select(PPGRecording)
        .where(PPGRecording.user_id == user_id, PPGRecording.started_at >= start, PPGRecording.started_at <= end)
        .order_by(PPGRecording.started_at)
    )
    return [PPGContract(started_at=r.started_at, fps=r.fps, **r.data) for r in db.scalars(q)]


def load_self_reports(db: Session, user_id: uuid.UUID, start: datetime, end: datetime) -> list[SelfReportContract]:
    q = (
        select(SelfReport)
        .where(SelfReport.user_id == user_id, SelfReport.at >= start, SelfReport.at <= end)
        .order_by(SelfReport.at)
    )
    return [
        SelfReportContract(at=r.at, mood=r.mood, stress=r.stress, energy=r.energy, focus=r.focus, note=r.note)
        for r in db.scalars(q)
    ]
