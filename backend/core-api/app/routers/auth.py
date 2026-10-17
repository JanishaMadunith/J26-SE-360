from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..config import Settings
from ..deps import get_db, get_settings_dep
from ..models import User
from ..schemas import LoginRequest, RefreshRequest, RegisterRequest, TokenResponse, UserOut
from ..security import ACCESS, REFRESH, TokenError, create_token, decode_token, hash_password, verify_password

router = APIRouter(prefix="/v1/auth", tags=["auth"])


def _tokens(user: User, settings: Settings) -> TokenResponse:
    return TokenResponse(
        access_token=create_token(user.id, ACCESS, settings),
        refresh_token=create_token(user.id, REFRESH, settings),
        expires_in=settings.access_token_minutes * 60,
        user=UserOut.model_validate(user),
    )


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest, db: Session = Depends(get_db), settings: Settings = Depends(get_settings_dep)):
    if db.scalar(select(User).where(User.email == body.email)) is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "An account with this email already exists")
    user = User(
        email=body.email,
        password_hash=hash_password(body.password),
        display_name=body.display_name,
        timezone=body.timezone,
    )
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "An account with this email already exists") from None
    db.refresh(user)
    return _tokens(user, settings)


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db), settings: Settings = Depends(get_settings_dep)):
    user = db.scalar(select(User).where(User.email == body.email))
    if user is None or not user.is_active or not verify_password(body.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect email or password")
    return _tokens(user, settings)


@router.post("/refresh", response_model=TokenResponse)
def refresh(body: RefreshRequest, db: Session = Depends(get_db), settings: Settings = Depends(get_settings_dep)):
    try:
        user_id = decode_token(body.refresh_token, REFRESH, settings)
    except TokenError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired refresh token") from None
    user = db.get(User, user_id)
    if user is None or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired refresh token")
    return _tokens(user, settings)
