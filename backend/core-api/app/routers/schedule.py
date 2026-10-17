from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from wellness_contracts import ScheduleResult

from ..config import Settings
from ..db import utcnow
from ..deps import get_current_user, get_db, get_module_clients, get_settings_dep
from ..models import User
from ..services.module_clients import ModuleClients, ModuleError
from ..services.pipeline import current_schedule, latest_wellness, run_schedule_step
from ..services.tasks import mark_overdue_missed

router = APIRouter(prefix="/v1/schedule", tags=["schedule"])


@router.get("/current", response_model=ScheduleResult)
def get_current(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = current_schedule(db, user.id)
    if s is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No schedule yet — add tasks and run the pipeline")
    return s


@router.post("/regenerate", response_model=ScheduleResult)
def regenerate(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    clients: ModuleClients = Depends(get_module_clients),
    settings: Settings = Depends(get_settings_dep),
):
    """Re-run Module 3 only, using the latest stored wellness forecast (fast path after the user edits tasks)."""
    now = utcnow()
    mark_overdue_missed(db, user.id, now)
    try:
        result = run_schedule_step(db, user, clients, settings, now, latest_wellness(db, user.id), "manual")
    except ModuleError as e:
        db.rollback()
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"Scheduler unavailable: {e}") from None
    db.commit()
    return result
