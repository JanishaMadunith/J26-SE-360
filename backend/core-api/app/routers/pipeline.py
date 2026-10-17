from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import Settings
from ..deps import get_current_user, get_db, get_module_clients, get_settings_dep
from ..models import PipelineRun, User
from ..schemas import PipelineRunOut
from ..services.module_clients import ModuleClients
from ..services.pipeline import run_pipeline

router = APIRouter(prefix="/v1/pipeline", tags=["pipeline"])


@router.post("/run", response_model=PipelineRunOut)
def run(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    clients: ModuleClients = Depends(get_module_clients),
    settings: Settings = Depends(get_settings_dep),
):
    """Run M1 -> M2 -> M3 -> M4 now for the current user (synchronous; useful for development and demos)."""
    return run_pipeline(db, user, clients, settings, trigger="manual")


@router.get("/runs", response_model=list[PipelineRunOut])
def runs(
    limit: int = Query(default=20, ge=1, le=200), user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    q = select(PipelineRun).where(PipelineRun.user_id == user.id).order_by(PipelineRun.started_at.desc()).limit(limit)
    return db.scalars(q).all()
