"""API-only schemas (everything exchanged with modules or the phone lives in `wellness_contracts`)."""

import re
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

from wellness_contracts import AffectiveStateResult, StepStatus, WellnessForecast

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _check_email(v: str) -> str:
    v = v.strip().lower()
    if not EMAIL_RE.match(v) or len(v) > 320:
        raise ValueError("invalid email address")
    return v


class RegisterRequest(_Request):
    email: str
    password: str = Field(min_length=8, max_length=72)
    display_name: str = Field(min_length=1, max_length=100)
    timezone: str = "Asia/Colombo"

    _email = field_validator("email")(_check_email)

    @field_validator("password")
    @classmethod
    def _password_bytes(cls, v: str) -> str:
        if len(v.encode("utf-8")) > 72:
            raise ValueError("password must be at most 72 bytes")
        return v

    @field_validator("timezone")
    @classmethod
    def _tz(cls, v: str) -> str:
        try:
            ZoneInfo(v)
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError(f"unknown timezone: {v}") from None
        return v


class LoginRequest(_Request):
    email: str
    password: str

    _email = field_validator("email")(_check_email)


class RefreshRequest(_Request):
    refresh_token: str


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str
    display_name: str
    timezone: str
    created_at: datetime
    consent_version: str | None = None
    consented_at: datetime | None = None


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="Access-token lifetime in seconds")
    user: UserOut


class ConsentRequest(_Request):
    version: str = Field(min_length=1, max_length=32)


class PipelineRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    trigger: str
    status: str
    steps: list[StepStatus]
    started_at: AwareDatetime
    finished_at: AwareDatetime | None = None


class InsightsOut(BaseModel):
    affective: AffectiveStateResult | None = None
    wellness: WellnessForecast | None = None


class HealthOut(BaseModel):
    status: str
    db: bool
    modules: dict[str, bool]
