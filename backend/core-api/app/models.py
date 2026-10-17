"""ORM models. One table per concept; module outputs are stored as the full contract JSON (`payload`)
so every result can be re-read exactly as the module produced it."""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base, JSONType, UTCDateTime, utcnow


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(100))
    display_name: Mapped[str] = mapped_column(String(100))
    timezone: Mapped[str] = mapped_column(String(64), default="Asia/Colombo")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    consent_version: Mapped[str | None] = mapped_column(String(32))
    consented_at: Mapped[datetime | None] = mapped_column(UTCDateTime)


class Device(Base):
    __tablename__ = "devices"
    __table_args__ = (UniqueConstraint("user_id", "device_id", name="uq_devices_user_device"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    device_id: Mapped[str] = mapped_column(String(64))
    app_version: Mapped[str] = mapped_column(String(32))
    first_seen: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    last_seen: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class SensorBatch(Base):
    """One phone upload, kept verbatim for traceability (`payload`)."""

    __tablename__ = "sensor_batches"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)  # = client batch_id (idempotency key)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    device_id: Mapped[str] = mapped_column(String(64))
    app_version: Mapped[str] = mapped_column(String(32))
    timezone: Mapped[str] = mapped_column(String(64))
    collected_from: Mapped[datetime] = mapped_column(UTCDateTime)
    collected_to: Mapped[datetime] = mapped_column(UTCDateTime)
    received_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    counts: Mapped[dict] = mapped_column(JSONType, default=dict)
    payload: Mapped[dict] = mapped_column(JSONType)


class SensorEvent(Base):
    """Long-format time series of small sensor events (keystroke, screen, app_usage, activity, steps, light).
    Composite primary key (id, ts) so the table can be turned into a TimescaleDB hypertable."""

    __tablename__ = "sensor_events"
    __table_args__ = (Index("ix_sensor_events_user_kind_ts", "user_id", "kind", "ts"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    ts: Mapped[datetime] = mapped_column(UTCDateTime, primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    batch_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("sensor_batches.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(32))
    value: Mapped[dict] = mapped_column(JSONType)


class PPGRecording(Base):
    __tablename__ = "ppg_recordings"
    __table_args__ = (Index("ix_ppg_user_started", "user_id", "started_at"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    batch_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("sensor_batches.id", ondelete="CASCADE"), index=True)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime)
    fps: Mapped[float] = mapped_column(Float)
    n_frames: Mapped[int] = mapped_column(Integer)
    data: Mapped[dict] = mapped_column(JSONType)  # {t_ms, red, green, blue, torch_on}


class SelfReport(Base):
    __tablename__ = "self_reports"
    __table_args__ = (Index("ix_self_reports_user_at", "user_id", "at"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    batch_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("sensor_batches.id", ondelete="SET NULL"))
    at: Mapped[datetime] = mapped_column(UTCDateTime)
    mood: Mapped[int | None] = mapped_column(Integer)
    stress: Mapped[int | None] = mapped_column(Integer)
    energy: Mapped[int | None] = mapped_column(Integer)
    focus: Mapped[int | None] = mapped_column(Integer)
    note: Mapped[str | None] = mapped_column(String(280))


class DietProfile(Base):
    __tablename__ = "diet_profiles"

    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    data: Mapped[dict] = mapped_column(JSONType)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)


class Task(Base):
    __tablename__ = "tasks"
    __table_args__ = (Index("ix_tasks_user_status", "user_id", "status"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    title: Mapped[str] = mapped_column(String(200))
    category: Mapped[str] = mapped_column(String(32))
    estimated_minutes: Mapped[int] = mapped_column(Integer)
    deadline: Mapped[datetime | None] = mapped_column(UTCDateTime)
    priority: Mapped[int] = mapped_column(Integer, default=3)
    fixed_start: Mapped[datetime | None] = mapped_column(UTCDateTime)
    fixed_end: Mapped[datetime | None] = mapped_column(UTCDateTime)
    notes: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), default="pending")
    times_rescheduled: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime)

    events: Mapped[list["TaskEvent"]] = relationship(
        back_populates="task", cascade="all, delete-orphan", order_by="TaskEvent.at"
    )


class TaskEvent(Base):
    """Append-only log of everything that happens to a task — Module 3's future training data."""

    __tablename__ = "task_events"
    __table_args__ = (Index("ix_task_events_user_at", "user_id", "at"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    task_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    event: Mapped[str] = mapped_column(String(16))
    at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    data: Mapped[dict] = mapped_column(JSONType, default=dict)

    task: Mapped[Task] = relationship(back_populates="events")


class _ModuleOutput:
    """Columns shared by stored module results."""

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    generated_at: Mapped[datetime] = mapped_column(UTCDateTime)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    model_name: Mapped[str] = mapped_column(String(64))
    model_version: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict] = mapped_column(JSONType)


class AffectiveState(_ModuleOutput, Base):
    __tablename__ = "affective_states"
    __table_args__ = (Index("ix_affective_user_generated", "user_id", "generated_at"),)

    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    window_start: Mapped[datetime] = mapped_column(UTCDateTime)
    window_end: Mapped[datetime] = mapped_column(UTCDateTime)
    label: Mapped[str] = mapped_column(String(16))


class WellnessForecast(_ModuleOutput, Base):
    __tablename__ = "wellness_forecasts"
    __table_args__ = (Index("ix_wellness_user_generated", "user_id", "generated_at"),)

    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))


class Schedule(_ModuleOutput, Base):
    __tablename__ = "schedules"
    __table_args__ = (Index("ix_schedules_user_current", "user_id", "is_current"),)

    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    horizon_start: Mapped[datetime] = mapped_column(UTCDateTime)
    horizon_end: Mapped[datetime] = mapped_column(UTCDateTime)
    reason: Mapped[str] = mapped_column(String(64))
    is_current: Mapped[bool] = mapped_column(Boolean, default=True)


class Recommendation(_ModuleOutput, Base):
    __tablename__ = "recommendations"
    __table_args__ = (Index("ix_recommendations_user_generated", "user_id", "generated_at"),)

    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"
    __table_args__ = (Index("ix_pipeline_runs_user_started", "user_id", "started_at"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    trigger: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16), default="running")
    started_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    steps: Mapped[list] = mapped_column(JSONType, default=list)
