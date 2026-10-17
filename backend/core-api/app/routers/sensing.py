from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from wellness_contracts import SelfReport, SensorBatch, SensorBatchAck

from ..config import Settings
from ..deps import get_current_user, get_db, get_settings_dep
from ..models import User
from ..services.pipeline import run_pipeline_background, should_auto_run
from ..services.sensing import BatchOwnershipError, ingest_batch, store_self_report

router = APIRouter(prefix="/v1", tags=["sensing"])


@router.post("/sensing/batches", response_model=SensorBatchAck, status_code=status.HTTP_202_ACCEPTED)
def upload_batch(
    body: SensorBatch,
    request: Request,
    background: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings_dep),
):
    """Store one raw sensor batch from the phone. Re-sending the same `batch_id` is safe (duplicate=true).
    If the user's pipeline has not run recently, it is started in the background."""
    try:
        ack = ingest_batch(db, user, body)
    except BatchOwnershipError:
        raise HTTPException(status.HTTP_409_CONFLICT, "batch_id already used") from None
    if not ack.duplicate and should_auto_run(db, user.id, settings):
        background.add_task(
            run_pipeline_background,
            request.app.state.session_factory,
            user.id,
            settings,
            request.app.state.module_transport,
        )
    return ack


@router.post("/self-reports", response_model=SelfReport, status_code=status.HTTP_201_CREATED)
def create_self_report(body: SelfReport, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return store_self_report(db, user, body)
