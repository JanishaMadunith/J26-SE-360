from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from wellness_contracts import DietProfile

from ..db import utcnow
from ..deps import get_current_user, get_db
from ..models import DietProfile as DietProfileRow
from ..models import User
from ..schemas import ConsentRequest, UserOut

router = APIRouter(prefix="/v1/me", tags=["me"])


@router.get("", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return user


@router.post("/consent", response_model=UserOut)
def consent(body: ConsentRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    user.consent_version = body.version
    user.consented_at = utcnow()
    db.commit()
    db.refresh(user)
    return user


@router.get("/diet-profile", response_model=DietProfile)
def get_diet_profile(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    row = db.get(DietProfileRow, user.id)
    return DietProfile.model_validate(row.data) if row else DietProfile()


@router.put("/diet-profile", response_model=DietProfile)
def put_diet_profile(body: DietProfile, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    row = db.get(DietProfileRow, user.id)
    data = body.model_dump(mode="json")
    if row is None:
        db.add(DietProfileRow(user_id=user.id, data=data))
    else:
        row.data = data
    db.commit()
    return body
