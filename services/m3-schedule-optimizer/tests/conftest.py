import uuid
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

COLOMBO = ZoneInfo("Asia/Colombo")
# Monday 19 Oct 2026, 08:00 Colombo
AS_OF = datetime(2026, 10, 19, 8, 0, tzinfo=COLOMBO)


def local(day_offset: int, hour: int, minute: int = 0) -> datetime:
    return (AS_OF + timedelta(days=day_offset)).replace(hour=hour, minute=minute)


def task(title="t", category="study", minutes=60, priority=3, deadline=None, fixed=None, rescheduled=0, tid=None):
    d = {
        "task_id": str(tid or uuid.uuid4()),
        "title": title,
        "category": category,
        "estimated_minutes": minutes,
        "priority": priority,
        "times_rescheduled": rescheduled,
    }
    if deadline is not None:
        d["deadline"] = deadline.isoformat()
    if fixed is not None:
        d["fixed_start"], d["fixed_end"] = fixed[0].isoformat(), fixed[1].isoformat()
    return d


def forecast(user_id, fn, hours=72, start=None):
    """Hourly forecast where fn(local_hour) -> (energy, focus, fatigue, burnout)."""
    start = start or AS_OF
    slots = []
    for h in range(hours):
        s = start + timedelta(hours=h)
        e, f, fa, b = fn(s.astimezone(COLOMBO).hour)
        slots.append(
            {
                "start": s.isoformat(),
                "end": (s + timedelta(hours=1)).isoformat(),
                "energy": e,
                "focus": f,
                "mental_fatigue": fa,
                "burnout_risk": b,
            }
        )
    return {
        "user_id": user_id,
        "generated_at": start.astimezone(UTC).isoformat(),
        "slot_minutes": 60,
        "slots": slots,
        "model": {"name": "test", "version": "0"},
    }


def request(tasks, busy=(), wellness=None, days=2, **kw):
    uid = str(uuid.uuid4())
    r = {
        "user_id": uid,
        "as_of": AS_OF.isoformat(),
        "timezone": "Asia/Colombo",
        "horizon_start": AS_OF.isoformat(),
        "horizon_end": (AS_OF + timedelta(days=days)).isoformat(),
        "tasks": tasks,
        "busy": [{"start": a.isoformat(), "end": b.isoformat()} for a, b in busy],
    }
    if wellness is not None:
        r["wellness"] = wellness(uid) if callable(wellness) else wellness
    r.update(kw)
    return r


@pytest.fixture
def helpers():
    class H:
        pass

    h = H()
    h.task, h.request, h.forecast, h.local, h.AS_OF = task, request, forecast, local, AS_OF
    return h
