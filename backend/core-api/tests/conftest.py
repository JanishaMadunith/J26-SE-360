"""Test fixtures.

Default: in-memory SQLite (fast). To run against PostgreSQL, migrate a database first and set
TEST_DATABASE_URL (deliberately NOT DATABASE_URL, so tests can never wipe your dev database):

    alembic upgrade head            # with DATABASE_URL pointing at the test database
    TEST_DATABASE_URL=postgresql+psycopg://... pytest
"""

import os
import uuid
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import Settings
from app.db import Base, build_engine
from app.main import create_app
from wellness_contracts import (
    AffectiveLabel,
    AffectiveStateRequest,
    AffectiveStateResult,
    FoodItem,
    MealRecommendation,
    MealType,
    ModelInfo,
    RecommendationRequest,
    Recommendations,
    ScheduledItem,
    ScheduleRequest,
    ScheduleResult,
    WellnessForecast,
    WellnessForecastRequest,
    WellnessSlot,
)

TEST_DB = os.environ.get("TEST_DATABASE_URL", "sqlite://")
MODULE_HOSTS = {"m1.test": "m1", "m2.test": "m2", "m3.test": "m3", "m4.test": "m4"}


class FakeModules:
    """Stand-ins for the four module services that return valid contract payloads.
    Add a module name to `down` to simulate an outage."""

    def __init__(self):
        self.down: set[str] = set()
        self.calls: list[str] = []
        self.last_requests: dict[str, dict] = {}

    def handler(self, request: httpx.Request) -> httpx.Response:
        name = MODULE_HOSTS[request.url.host]
        if name in self.down:
            raise httpx.ConnectError(f"{name} is down", request=request)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok", "service": name})
        self.calls.append(name)
        body = request.content
        if name == "m1":
            req = AffectiveStateRequest.model_validate_json(body)
            probs = {lab: 0.2 for lab in AffectiveLabel}
            out = AffectiveStateResult(
                user_id=req.user_id,
                window_start=req.window_start,
                window_end=req.window_end,
                label=AffectiveLabel.relaxed,
                probabilities=probs,
                confidence=0.4,
                features={"n_keystrokes": float(len(req.keystrokes))},
                model=ModelInfo(name="fake-m1", version="0"),
            )
        elif name == "m2":
            req = WellnessForecastRequest.model_validate_json(body)
            slots = [
                WellnessSlot(
                    start=req.as_of + timedelta(hours=h),
                    end=req.as_of + timedelta(hours=h + 1),
                    mental_fatigue=0.3,
                    focus=0.6,
                    energy=0.7,
                    burnout_risk=0.1,
                )
                for h in range(req.horizon_hours)
            ]
            out = WellnessForecast(
                user_id=req.user_id,
                generated_at=req.as_of,
                slot_minutes=60,
                slots=slots,
                model=ModelInfo(name="fake-m2", version="0"),
            )
        elif name == "m3":
            req = ScheduleRequest.model_validate_json(body)
            cursor = req.as_of + timedelta(hours=1)
            items = []
            for t in req.tasks:
                if t.fixed_start is not None:
                    items.append(
                        ScheduledItem(
                            task_id=t.task_id,
                            start=t.fixed_start,
                            end=t.fixed_end,
                            completion_probability=0.9,
                            is_fixed=True,
                        )
                    )
                    continue
                end = cursor + timedelta(minutes=t.estimated_minutes)
                items.append(ScheduledItem(task_id=t.task_id, start=cursor, end=end, completion_probability=0.8))
                cursor = end
            out = ScheduleResult(
                user_id=req.user_id,
                generated_at=req.as_of,
                horizon_start=req.horizon_start,
                horizon_end=req.horizon_end,
                items=items,
                reason=req.reason,
                model=ModelInfo(name="fake-m3", version="0"),
            )
        else:
            req = RecommendationRequest.model_validate_json(body)
            meal = MealRecommendation(
                meal_type=MealType.lunch,
                window_start=req.as_of,
                window_end=req.as_of + timedelta(hours=1),
                foods=[FoodItem(name="Rice and dhal", portion="1 plate", reason="steady energy")],
                rationale="test",
            )
            out = Recommendations(
                user_id=req.user_id, generated_at=req.as_of, meals=[meal], model=ModelInfo(name="fake-m4", version="0")
            )
        self.last_requests[name] = request.read().decode()
        return httpx.Response(200, content=out.model_dump_json(), headers={"Content-Type": "application/json"})


@pytest.fixture
def settings() -> Settings:
    return Settings(
        database_url=TEST_DB,
        redis_url=None,
        jwt_secret="test-secret-that-is-at-least-32-bytes-long",
        m1_url="http://m1.test",
        m2_url="http://m2.test",
        m3_url="http://m3.test",
        m4_url="http://m4.test",
        pipeline_min_interval_minutes=15,
        pipeline_window_minutes=60,
    )


@pytest.fixture
def engine(settings):
    eng = build_engine(TEST_DB)
    if eng.dialect.name == "sqlite":
        Base.metadata.create_all(eng)
    else:  # schema comes from Alembic; just empty every table
        names = ", ".join(t.name for t in Base.metadata.sorted_tables)
        with eng.begin() as conn:
            conn.execute(text(f"TRUNCATE TABLE {names} CASCADE"))
    yield eng
    eng.dispose()


@pytest.fixture
def fake_modules() -> FakeModules:
    return FakeModules()


@pytest.fixture
def client(settings, engine, fake_modules):
    app = create_app(settings=settings, engine=engine, module_transport=httpx.MockTransport(fake_modules.handler))
    with TestClient(app) as c:
        yield c


@pytest.fixture
def db(engine):
    with Session(engine) as s:
        yield s


def register(client, email: str | None = None, password: str = "correct-horse-1") -> dict:
    email = email or f"user-{uuid.uuid4().hex[:8]}@example.com"
    r = client.post("/v1/auth/register", json={"email": email, "password": password, "display_name": "Test User"})
    assert r.status_code == 201, r.text
    return r.json()


def auth_headers(tokens: dict) -> dict:
    return {"Authorization": f"Bearer {tokens['access_token']}"}


@pytest.fixture
def user(client) -> dict:
    tokens = register(client)
    return {"tokens": tokens, "headers": auth_headers(tokens), "id": tokens["user"]["id"]}


def now_utc() -> datetime:
    return datetime.now(UTC)


def iso(d: datetime) -> str:
    return d.isoformat()


def sample_batch(minutes_ago: int = 5, batch_id: str | None = None) -> dict:
    """A realistic batch containing every signal kind."""
    t = now_utc() - timedelta(minutes=minutes_ago)
    return {
        "batch_id": batch_id or str(uuid.uuid4()),
        "device_id": "pixel-7a-test",
        "app_version": "0.1.0",
        "timezone": "Asia/Colombo",
        "collected_from": iso(t - timedelta(minutes=15)),
        "collected_to": iso(t),
        "ppg": [
            {
                "started_at": iso(t - timedelta(minutes=1)),
                "fps": 30.0,
                "t_ms": [0, 33, 66, 100],
                "red": [200.0, 201.5, 199.0, 200.2],
                "green": [50.0, 51.0, 49.5, 50.1],
                "blue": [20.0, 20.5, 21.0, 20.2],
                "torch_on": True,
            }
        ],
        "keystrokes": [
            {"down_at": iso(t - timedelta(seconds=10 - i)), "hold_ms": 80 + i, "key_class": "letter"} for i in range(5)
        ],
        "screen_events": [
            {"at": iso(t - timedelta(minutes=10)), "type": "unlock"},
            {"at": iso(t - timedelta(minutes=2)), "type": "screen_off"},
        ],
        "app_usage": [
            {
                "start": iso(t - timedelta(minutes=9)),
                "end": iso(t - timedelta(minutes=3)),
                "category": "social",
                "app_hash": "a1b2c3d4e5f6",
            }
        ],
        "activity": [{"at": iso(t - timedelta(minutes=12)), "activity": "walking", "entering": True}],
        "steps": [{"start": iso(t - timedelta(minutes=15)), "end": iso(t), "steps": 420}],
        "light": [{"at": iso(t - timedelta(minutes=4)), "lux": 180.0}],
        "self_reports": [{"at": iso(t - timedelta(minutes=1)), "mood": 4, "stress": 2, "energy": 3, "focus": 4}],
    }
