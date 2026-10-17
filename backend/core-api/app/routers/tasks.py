import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from wellness_contracts import Task, TaskCreate, TaskEventType, TaskStatus, TaskUpdate

from ..db import utcnow
from ..deps import get_current_user, get_db
from ..models import Task as TaskRow
from ..models import User
from ..services.tasks import log_event, to_contract

router = APIRouter(prefix="/v1/tasks", tags=["tasks"])

STATUS_EVENTS = {
    TaskStatus.done: TaskEventType.completed,
    TaskStatus.missed: TaskEventType.missed,
    TaskStatus.cancelled: TaskEventType.cancelled,
    TaskStatus.scheduled: TaskEventType.scheduled,
    TaskStatus.pending: TaskEventType.rescheduled,
}


def _get_owned(db: Session, user: User, task_id: uuid.UUID) -> TaskRow:
    t = db.get(TaskRow, task_id)
    if t is None or t.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Task not found")
    return t


@router.get("", response_model=list[Task])
def list_tasks(
    status_filter: TaskStatus | None = Query(default=None, alias="status"),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List the user's tasks (earliest deadline first). Filter with ?status=pending|scheduled|done|missed|cancelled."""
    q = select(TaskRow).where(TaskRow.user_id == user.id)
    if status_filter is not None:
        q = q.where(TaskRow.status == status_filter.value)
    rows = db.scalars(q.order_by(TaskRow.created_at)).all()
    far = utcnow().replace(year=9999)
    rows = sorted(rows, key=lambda t: (t.deadline or far, t.created_at))
    return [to_contract(t) for t in rows]


@router.post("", response_model=Task, status_code=status.HTTP_201_CREATED)
def create_task(body: TaskCreate, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    t = TaskRow(
        user_id=user.id,
        title=body.title,
        category=body.category.value,
        estimated_minutes=body.estimated_minutes,
        deadline=body.deadline,
        priority=body.priority,
        fixed_start=body.fixed_start,
        fixed_end=body.fixed_end,
        notes=body.notes,
        status=TaskStatus.pending.value,
    )
    db.add(t)
    db.flush()
    log_event(db, t, TaskEventType.created, {"deadline": body.deadline.isoformat() if body.deadline else None})
    db.commit()
    db.refresh(t)
    return to_contract(t)


@router.patch("/{task_id}", response_model=Task)
def update_task(
    task_id: uuid.UUID, body: TaskUpdate, user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    t = _get_owned(db, user, task_id)
    changes = body.model_dump(exclude_unset=True)
    new_status = changes.pop("status", None)
    for field, value in changes.items():
        setattr(t, field, value.value if hasattr(value, "value") else value)
    if new_status is not None and new_status.value != t.status:
        t.status = new_status.value
        if new_status == TaskStatus.done:
            t.completed_at = utcnow()
        log_event(db, t, STATUS_EVENTS[new_status], {"via": "update"})
    db.commit()
    db.refresh(t)
    return to_contract(t)


@router.post("/{task_id}/complete", response_model=Task)
def complete_task(task_id: uuid.UUID, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    t = _get_owned(db, user, task_id)
    if t.status != TaskStatus.done.value:
        now = utcnow()
        t.status = TaskStatus.done.value
        t.completed_at = now
        on_time = t.deadline is None or now <= t.deadline
        log_event(db, t, TaskEventType.completed, {"on_time": on_time}, at=now)
        db.commit()
        db.refresh(t)
    return to_contract(t)


@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_task(task_id: uuid.UUID, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Soft delete: the task becomes 'cancelled' so its history stays available for research."""
    t = _get_owned(db, user, task_id)
    if t.status != TaskStatus.cancelled.value:
        t.status = TaskStatus.cancelled.value
        log_event(db, t, TaskEventType.cancelled)
        db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
