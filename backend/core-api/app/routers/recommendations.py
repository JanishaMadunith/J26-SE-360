from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from wellness_contracts import Recommendations

from ..deps import get_current_user, get_db
from ..models import User
from ..services.pipeline import latest_recommendations

router = APIRouter(prefix="/v1/recommendations", tags=["recommendations"])


@router.get("/latest", response_model=Recommendations)
def latest(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Latest meal recommendations and suggested activities (Module 4)."""
    r = latest_recommendations(db, user.id)
    if r is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No recommendations yet")
    return r
