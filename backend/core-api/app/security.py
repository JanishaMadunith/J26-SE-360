"""Password hashing (bcrypt) and JWT access/refresh tokens."""

import uuid
from datetime import UTC, datetime, timedelta

import bcrypt
import jwt

from .config import Settings

ACCESS = "access"
REFRESH = "refresh"


class TokenError(Exception):
    pass


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("ascii"))
    except ValueError:
        return False


def create_token(user_id: uuid.UUID, token_type: str, settings: Settings) -> str:
    now = datetime.now(UTC)
    ttl = (
        timedelta(minutes=settings.access_token_minutes)
        if token_type == ACCESS
        else timedelta(days=settings.refresh_token_days)
    )
    claims = {"sub": str(user_id), "type": token_type, "iat": now, "exp": now + ttl, "jti": uuid.uuid4().hex}
    return jwt.encode(claims, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str, expected_type: str, settings: Settings) -> uuid.UUID:
    try:
        claims = jwt.decode(
            token, settings.jwt_secret, algorithms=[settings.jwt_algorithm], options={"require": ["sub", "type", "exp"]}
        )
    except jwt.PyJWTError as e:
        raise TokenError(str(e)) from e
    if claims.get("type") != expected_type:
        raise TokenError(f"expected a {expected_type} token")
    try:
        return uuid.UUID(claims["sub"])
    except ValueError as e:
        raise TokenError("bad subject") from e
