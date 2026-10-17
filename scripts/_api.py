"""Tiny helpers shared by the demo scripts (talk to the core API with httpx)."""

from __future__ import annotations

import sys

import httpx

DEMO_EMAIL = "demo@j26se360.lk"
DEMO_PASSWORD = "demo-pass-2026"
DEMO_NAME = "Demo Student"


def fail(msg: str, r: httpx.Response | None = None) -> None:
    detail = f" [{r.status_code}] {r.text[:300]}" if r is not None else ""
    print(f"ERROR: {msg}{detail}", file=sys.stderr)
    sys.exit(1)


def client(base_url: str) -> httpx.Client:
    c = httpx.Client(base_url=base_url.rstrip("/"), timeout=60)
    try:
        c.get("/health").raise_for_status()
    except httpx.HTTPError as e:
        fail(f"core API not reachable at {base_url} ({e}). Is the stack running? Try `make up`.")
    return c


def login_or_register(
    c: httpx.Client,
    email: str = DEMO_EMAIL,
    password: str = DEMO_PASSWORD,
    name: str = DEMO_NAME,
    tz: str = "Asia/Colombo",
) -> dict:
    """Log in (registering first if needed) and attach the bearer token to the client."""
    r = c.post("/v1/auth/login", json={"email": email, "password": password})
    if r.status_code != 200:
        r = c.post(
            "/v1/auth/register", json={"email": email, "password": password, "display_name": name, "timezone": tz}
        )
        if r.status_code not in (200, 201):
            fail("could not register or log in the demo user", r)
    tokens = r.json()
    c.headers["Authorization"] = f"Bearer {tokens['access_token']}"
    return tokens


def demo_tasks(now_local) -> list[dict]:
    """A realistic student task list relative to `now_local` (aware datetime in the user's timezone)."""
    from datetime import timedelta

    day = now_local.replace(minute=0, second=0, microsecond=0)
    tomorrow = day + timedelta(days=1)
    return [
        {
            "title": "Read 3 papers for the literature review",
            "category": "study",
            "estimated_minutes": 150,
            "priority": 5,
            "deadline": (day + timedelta(days=2)).replace(hour=18).isoformat(),
        },
        {
            "title": "Implement Module 3 scheduler tests",
            "category": "work",
            "estimated_minutes": 90,
            "priority": 4,
            "deadline": tomorrow.replace(hour=17).isoformat(),
        },
        {"title": "Gym session", "category": "exercise", "estimated_minutes": 60, "priority": 3},
        {
            "title": "Laundry",
            "category": "chores",
            "estimated_minutes": 30,
            "priority": 2,
            "deadline": (day + timedelta(days=3)).replace(hour=21).isoformat(),
        },
        {
            "title": "Research methods lecture",
            "category": "study",
            "estimated_minutes": 120,
            "priority": 4,
            "fixed_start": tomorrow.replace(hour=10).isoformat(),
            "fixed_end": tomorrow.replace(hour=12).isoformat(),
        },
    ]


def ensure_demo_tasks(c: httpx.Client, now_local) -> list[dict]:
    r = c.get("/v1/tasks")
    if r.status_code != 200:
        fail("could not list tasks", r)
    existing = [t for t in r.json() if t.get("status") in ("pending", "scheduled")]
    if existing:
        return existing
    created = []
    for t in demo_tasks(now_local):
        r = c.post("/v1/tasks", json=t)
        if r.status_code not in (200, 201):
            fail(f"could not create task {t['title']!r}", r)
        created.append(r.json())
    return created
