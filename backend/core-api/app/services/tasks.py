"""Task helpers: the event log and automatic 'missed' detection."""

import uuid
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from wellness_contracts import Task as TaskContract
from wellness_contracts import TaskEventType, TaskStatus

from ..db import utcnow
from ..models import Task, TaskEvent

OPEN_STATUSES = (TaskStatus.pending.value, TaskStatus.scheduled.value)


def log_event(
    db: Session, task: Task, event: TaskEventType, data: dict | None = None, at: datetime | None = None
) -> None:
    db.add(TaskEvent(task_id=task.id, user_id=task.user_id, event=event.value, at=at or utcnow(), data=data or {}))


def to_contract(t: Task) -> TaskContract:
    return TaskContract(
        id=t.id,
        title=t.title,
        category=t.category,
        estimated_minutes=t.estimated_minutes,
        deadline=t.deadline,
        priority=t.priority,
        fixed_start=t.fixed_start,
        fixed_end=t.fixed_end,
        notes=t.notes,
        status=t.status,
        created_at=t.created_at,
        completed_at=t.completed_at,
    )


def mark_overdue_missed(db: Session, user_id: uuid.UUID, now: datetime) -> int:
    """Open tasks whose deadline has passed become 'missed' (logged). Returns how many changed."""
    q = select(Task).where(
        Task.user_id == user_id, Task.status.in_(OPEN_STATUSES), Task.deadline.is_not(None), Task.deadline < now
    )
    n = 0
    for t in db.scalars(q):
        t.status = TaskStatus.missed.value
        log_event(db, t, TaskEventType.missed, {"deadline": t.deadline.isoformat()}, at=now)
        n += 1
    return n
