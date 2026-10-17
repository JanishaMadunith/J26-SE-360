from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..deps import get_current_user, get_db
from ..models import User
from ..schemas import InsightsOut
from ..services.pipeline import latest_affective, latest_wellness

router = APIRouter(prefix="/v1/insights", tags=["insights"])


@router.get("/latest", response_model=InsightsOut)
def insights(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Latest affective state (Module 1) and wellness forecast (Module 2). Either may be null early on."""
    return InsightsOut(affective=latest_affective(db, user.id), wellness=latest_wellness(db, user.id))
