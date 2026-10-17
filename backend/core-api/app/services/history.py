"""User behaviour summary for Module 3, computed from tasks and the task-event log."""

import uuid
from collections import defaultdict
from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from wellness_contracts import TaskStatus, UserHistory

from ..models import Task, TaskEvent


def compute_user_history(db: Session, user_id: uuid.UUID, now: datetime, days: int = 30) -> UserHistory:
    since = now - timedelta(days=days)
    completed = (
        db.scalar(
            select(func.count())
            .select_from(Task)
            .where(Task.user_id == user_id, Task.status == TaskStatus.done.value, Task.completed_at >= since)
        )
        or 0
    )
    missed = (
        db.scalar(
            select(func.count())
            .select_from(TaskEvent)
            .where(TaskEvent.user_id == user_id, TaskEvent.event == "missed", TaskEvent.at >= since)
        )
        or 0
    )

    # on-time rate over tasks whose deadline fell in the window and whose outcome is known
    q = select(Task.category, Task.status, Task.completed_at, Task.deadline).where(
        Task.user_id == user_id,
        Task.deadline.is_not(None),
        Task.deadline >= since,
        Task.deadline <= now,
        Task.status.in_((TaskStatus.done.value, TaskStatus.missed.value)),
    )
    total, on_time = 0, 0
    by_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for cat, status, completed_at, deadline in db.execute(q):
        ok = int(status == TaskStatus.done.value and completed_at is not None and completed_at <= deadline)
        total += 1
        on_time += ok
        by_cat[cat][0] += ok
        by_cat[cat][1] += 1
    return UserHistory(
        tasks_completed_30d=completed,
        tasks_missed_30d=missed,
        on_time_rate_30d=(on_time / total) if total else None,
        on_time_rate_by_category={c: v[0] / v[1] for c, v in by_cat.items() if v[1]},
    )
