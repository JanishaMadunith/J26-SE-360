"""Pipeline orchestrator: M1 -> M2 -> M3 -> M4 for one user.

Each step is recorded as a StepStatus. If a module fails, the last stored result is used instead
(`used_fallback=True`) and the run continues, so one broken module never blocks the others.
"""

import logging
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session, sessionmaker

from wellness_contracts import (
    AffectiveStateRequest,
    AffectiveStateResult,
    DietProfile,
    PipelineStep,
    RecommendationRequest,
    Recommendations,
    SchedulableTask,
    ScheduleRequest,
    ScheduleResult,
    StepStatus,
    TaskEventType,
    TaskStatus,
    WellnessForecast,
    WellnessForecastRequest,
    WorkloadSummary,
)

from .. import models as m
from ..config import Settings
from ..db import utcnow
from .history import compute_user_history
from .module_clients import ModuleClients, ModuleError
from .sensing import load_events, load_ppg, load_self_reports
from .tasks import OPEN_STATUSES, log_event, mark_overdue_missed

log = logging.getLogger("core.pipeline")


# ---------------------------------------------------------------- latest stored results (fallbacks)
def latest_affective(db: Session, user_id: uuid.UUID) -> AffectiveStateResult | None:
    row = db.scalar(
        select(m.AffectiveState)
        .where(m.AffectiveState.user_id == user_id)
        .order_by(m.AffectiveState.generated_at.desc(), m.AffectiveState.created_at.desc())
        .limit(1)
    )
    return AffectiveStateResult.model_validate(row.payload) if row else None


def latest_wellness(db: Session, user_id: uuid.UUID) -> WellnessForecast | None:
    row = db.scalar(
        select(m.WellnessForecast)
        .where(m.WellnessForecast.user_id == user_id)
        .order_by(m.WellnessForecast.generated_at.desc(), m.WellnessForecast.created_at.desc())
        .limit(1)
    )
    return WellnessForecast.model_validate(row.payload) if row else None


def current_schedule(db: Session, user_id: uuid.UUID) -> ScheduleResult | None:
    row = db.scalar(
        select(m.Schedule)
        .where(m.Schedule.user_id == user_id, m.Schedule.is_current.is_(True))
        .order_by(m.Schedule.created_at.desc())
        .limit(1)
    )
    return ScheduleResult.model_validate(row.payload) if row else None


def latest_recommendations(db: Session, user_id: uuid.UUID) -> Recommendations | None:
    row = db.scalar(
        select(m.Recommendation)
        .where(m.Recommendation.user_id == user_id)
        .order_by(m.Recommendation.generated_at.desc(), m.Recommendation.created_at.desc())
        .limit(1)
    )
    return Recommendations.model_validate(row.payload) if row else None


def diet_profile(db: Session, user_id: uuid.UUID) -> DietProfile:
    row = db.get(m.DietProfile, user_id)
    return DietProfile.model_validate(row.data) if row else DietProfile()


# ---------------------------------------------------------------- storing results
def _store_affective(db: Session, r: AffectiveStateResult) -> None:
    db.add(
        m.AffectiveState(
            user_id=r.user_id,
            window_start=r.window_start,
            window_end=r.window_end,
            label=r.label.value,
            generated_at=r.window_end,
            model_name=r.model.name,
            model_version=r.model.version,
            payload=r.model_dump(mode="json"),
        )
    )


def _store_wellness(db: Session, r: WellnessForecast) -> None:
    db.add(
        m.WellnessForecast(
            user_id=r.user_id,
            generated_at=r.generated_at,
            model_name=r.model.name,
            model_version=r.model.version,
            payload=r.model_dump(mode="json"),
        )
    )


def _store_recommendations(db: Session, r: Recommendations) -> None:
    db.add(
        m.Recommendation(
            user_id=r.user_id,
            generated_at=r.generated_at,
            model_name=r.model.name,
            model_version=r.model.version,
            payload=r.model_dump(mode="json"),
        )
    )


def store_schedule(db: Session, user_id: uuid.UUID, r: ScheduleResult) -> None:
    """Make `r` the current schedule and update task statuses + the event log."""
    previous = current_schedule(db, user_id)
    prev_slots = {i.task_id: (i.start, i.end) for i in previous.items} if previous else {}
    db.execute(
        update(m.Schedule)
        .where(m.Schedule.user_id == user_id, m.Schedule.is_current.is_(True))
        .values(is_current=False)
    )
    db.add(
        m.Schedule(
            user_id=user_id,
            generated_at=r.generated_at,
            horizon_start=r.horizon_start,
            horizon_end=r.horizon_end,
            reason=r.reason,
            is_current=True,
            model_name=r.model.name,
            model_version=r.model.version,
            payload=r.model_dump(mode="json"),
        )
    )

    ids = {i.task_id for i in r.items} | {u.task_id for u in r.unscheduled}
    tasks = {t.id: t for t in db.scalars(select(m.Task).where(m.Task.user_id == user_id, m.Task.id.in_(ids)))}
    for item in r.items:
        t = tasks.get(item.task_id)
        if t is None or t.status not in OPEN_STATUSES:
            continue
        slot = {
            "start": item.start.isoformat(),
            "end": item.end.isoformat(),
            "completion_probability": item.completion_probability,
            "reason": r.reason,
        }
        old = prev_slots.get(item.task_id)
        if t.status == TaskStatus.scheduled.value and old is not None:
            if old != (item.start, item.end):
                t.times_rescheduled += 1
                log_event(db, t, TaskEventType.rescheduled, {**slot, "previous_start": old[0].isoformat()})
        else:
            log_event(db, t, TaskEventType.scheduled, slot)
        t.status = TaskStatus.scheduled.value
    for u in r.unscheduled:
        t = tasks.get(u.task_id)
        if t is not None and t.status == TaskStatus.scheduled.value:
            t.status = TaskStatus.pending.value
            log_event(db, t, TaskEventType.rescheduled, {"unscheduled": u.reason})


# ---------------------------------------------------------------- request builders
def _workload(db: Session, user_id: uuid.UUID, now: datetime) -> WorkloadSummary:
    open_q = select(m.Task).where(m.Task.user_id == user_id, m.Task.status.in_(OPEN_STATUSES))
    due = [t for t in db.scalars(open_q) if t.deadline is not None and now < t.deadline <= now + timedelta(hours=24)]
    pending = (
        db.scalar(
            select(func.count()).select_from(m.Task).where(m.Task.user_id == user_id, m.Task.status.in_(OPEN_STATUSES))
        )
        or 0
    )
    overdue = (
        db.scalar(
            select(func.count())
            .select_from(m.Task)
            .where(
                m.Task.user_id == user_id,
                m.Task.status == TaskStatus.missed.value,
                m.Task.deadline.is_not(None),
                m.Task.deadline >= now - timedelta(hours=24),
            )
        )
        or 0
    )
    return WorkloadSummary(
        tasks_due_24h=len(due),
        minutes_due_24h=sum(t.estimated_minutes for t in due),
        tasks_overdue=overdue,
        tasks_pending=pending,
    )


def build_schedule_request(
    db: Session, user: m.User, now: datetime, settings: Settings, wellness: WellnessForecast | None, reason: str
) -> ScheduleRequest:
    q = select(m.Task).where(m.Task.user_id == user.id, m.Task.status.in_(OPEN_STATUSES))
    tasks = []
    for t in db.scalars(q):
        if t.fixed_end is not None and t.fixed_end < now:
            continue
        tasks.append(
            SchedulableTask(
                task_id=t.id,
                title=t.title,
                category=t.category,
                estimated_minutes=t.estimated_minutes,
                priority=t.priority,
                deadline=t.deadline,
                fixed_start=t.fixed_start,
                fixed_end=t.fixed_end,
                times_rescheduled=t.times_rescheduled,
            )
        )
    return ScheduleRequest(
        user_id=user.id,
        as_of=now,
        timezone=user.timezone,
        horizon_start=now,
        horizon_end=now + timedelta(days=settings.schedule_horizon_days),
        tasks=tasks,
        wellness=wellness,
        history=compute_user_history(db, user.id, now),
        reason=reason,
    )


# ---------------------------------------------------------------- steps
def _timed(step: PipelineStep, fn: Callable, fallback: Callable):
    """Run fn(); on ModuleError use fallback(). Returns (result, StepStatus)."""
    t0 = time.perf_counter()
    try:
        result = fn()
        return result, StepStatus(step=step, ok=True, ms=int((time.perf_counter() - t0) * 1000))
    except ModuleError as e:
        log.warning("pipeline step %s failed: %s", step.value, e)
        fb = fallback()
        return fb, StepStatus(
            step=step,
            ok=False,
            ms=int((time.perf_counter() - t0) * 1000),
            used_fallback=fb is not None,
            error=str(e)[:500],
        )


def run_schedule_step(
    db: Session,
    user: m.User,
    clients: ModuleClients,
    settings: Settings,
    now: datetime,
    wellness: WellnessForecast | None,
    reason: str,
) -> ScheduleResult:
    """Call M3 and store the result as the current schedule. Raises ModuleError on failure."""
    result = clients.schedule(build_schedule_request(db, user, now, settings, wellness, reason))
    store_schedule(db, user.id, result)
    return result


def run_pipeline(
    db: Session,
    user: m.User,
    clients: ModuleClients,
    settings: Settings,
    trigger: str = "manual",
    now: datetime | None = None,
) -> m.PipelineRun:
    now = now or utcnow()
    run = m.PipelineRun(user_id=user.id, trigger=trigger, status="running", started_at=now)
    db.add(run)
    mark_overdue_missed(db, user.id, now)
    db.flush()
    steps: list[StepStatus] = []

    # ---- M1: affective state from the last window of raw signals
    w_start = now - timedelta(minutes=settings.pipeline_window_minutes)
    signals = load_events(db, user.id, w_start, now)
    ppg = load_ppg(db, user.id, w_start, now)
    if not ppg and not any(signals.values()):
        affective = latest_affective(db, user.id)
        steps.append(
            StepStatus(
                step=PipelineStep.affective,
                ok=True,
                ms=0,
                used_fallback=affective is not None,
                error="skipped: no sensor data in window",
            )
        )
    else:
        req = AffectiveStateRequest(
            user_id=user.id,
            window_start=w_start,
            window_end=now,
            timezone=user.timezone,
            ppg=ppg,
            self_reports=load_self_reports(db, user.id, w_start, now),
            **signals,
        )

        def _m1():
            r = clients.affective_state(req)
            _store_affective(db, r)
            return r

        affective, st = _timed(PipelineStep.affective, _m1, lambda: latest_affective(db, user.id))
        steps.append(st)

    # ---- M2: wellness forecast
    hist_rows = db.scalars(
        select(m.AffectiveState)
        .where(m.AffectiveState.user_id == user.id, m.AffectiveState.generated_at >= now - timedelta(days=7))
        .order_by(m.AffectiveState.generated_at)
    )
    ctx = load_events(db, user.id, now - timedelta(hours=48), now, kinds=("screen", "activity", "steps", "light"))
    w_req = WellnessForecastRequest(
        user_id=user.id,
        as_of=now,
        timezone=user.timezone,
        affective_history=[AffectiveStateResult.model_validate(r.payload) for r in hist_rows],
        self_reports=load_self_reports(db, user.id, now - timedelta(days=7), now),
        workload=_workload(db, user.id, now),
        **ctx,
    )

    def _m2():
        r = clients.wellness_forecast(w_req)
        _store_wellness(db, r)
        return r

    wellness, st = _timed(PipelineStep.wellness, _m2, lambda: latest_wellness(db, user.id))
    steps.append(st)

    # ---- M3: schedule
    schedule, st = _timed(
        PipelineStep.schedule,
        lambda: run_schedule_step(db, user, clients, settings, now, wellness, "scheduled"),
        lambda: current_schedule(db, user.id),
    )
    steps.append(st)

    # ---- M4: recommendations
    r_req = RecommendationRequest(
        user_id=user.id,
        as_of=now,
        timezone=user.timezone,
        affective_state=affective,
        wellness=wellness,
        schedule=schedule,
        profile=diet_profile(db, user.id),
    )

    def _m4():
        r = clients.recommendations(r_req)
        _store_recommendations(db, r)
        return r

    _, st = _timed(PipelineStep.recommendations, _m4, lambda: latest_recommendations(db, user.id))
    steps.append(st)

    n_ok = sum(s.ok for s in steps)
    run.status = "ok" if n_ok == len(steps) else ("failed" if n_ok == 0 else "partial")
    run.steps = [s.model_dump(mode="json") for s in steps]
    run.finished_at = utcnow()
    db.commit()
    db.refresh(run)
    return run


def should_auto_run(db: Session, user_id: uuid.UUID, settings: Settings, now: datetime | None = None) -> bool:
    now = now or utcnow()
    last = db.scalar(select(func.max(m.PipelineRun.started_at)).where(m.PipelineRun.user_id == user_id))
    if last is None:
        return True
    if last.tzinfo is None:  # some drivers return naive values from aggregate functions
        last = last.replace(tzinfo=UTC)
    return now - last >= timedelta(minutes=settings.pipeline_min_interval_minutes)


def run_pipeline_background(
    session_factory: sessionmaker, user_id: uuid.UUID, settings: Settings, transport=None
) -> None:
    """Entry point for FastAPI BackgroundTasks — opens its own session and never raises."""
    db = session_factory()
    clients = ModuleClients(settings, transport=transport)
    try:
        user = db.get(m.User, user_id)
        if user is not None:
            run_pipeline(db, user, clients, settings, trigger="sensor_upload")
    except Exception:  # noqa: BLE001 — background job must not crash the worker
        log.exception("background pipeline failed for user %s", user_id)
        db.rollback()
    finally:
        clients.close()
        db.close()
