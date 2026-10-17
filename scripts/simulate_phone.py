"""Simulate the phone app end-to-end against a running core API.

1. log in / register the demo user and make sure they have tasks
2. upload one realistic RAW SensorBatch covering the last 24 h (camera PPG at ~72 bpm, keystroke timings,
   screen events incl. a night's sleep, app sessions, activity, steps, ambient light, a self-report)
3. run the M1 → M2 → M3 → M4 pipeline
4. print the affective state, wellness forecast, schedule and meal recommendations

  python scripts/simulate_phone.py [--base-url http://localhost:8000] [--seed 7]      (or: make simulate)
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import uuid
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from _api import client, ensure_demo_tasks, fail, login_or_register

from wellness_contracts import (
    ActivityTransition,
    AppUsageSession,
    KeystrokeEvent,
    LightReading,
    PPGRecording,
    ScreenEvent,
    SelfReport,
    SensorBatch,
    StepCount,
)

SALT = "j26se360-demo-salt"
APPS = [
    ("com.whatsapp", "social"),
    ("com.instagram.android", "social"),
    ("com.google.android.youtube", "video"),
    ("com.android.chrome", "productivity"),
    ("com.google.android.gm", "productivity"),
    ("com.spotify.music", "audio"),
    ("com.supercell.clashroyale", "game"),
    ("lk.sliit.courseweb", "unknown"),
]


def app_hash(pkg: str) -> str:
    return hashlib.sha256((SALT + pkg).encode()).hexdigest()[:16]


def synth_ppg(start: datetime, rng: random.Random, bpm: float = 72, seconds: int = 60, fps: float = 30.0):
    n = int(seconds * fps)
    t_ms, red, green, blue = [], [], [], []
    phase = 0.0
    for i in range(n):
        t = i / fps
        f = bpm / 60 * (1 + 0.04 * math.sin(2 * math.pi * 0.25 * t))  # respiratory-like HRV
        phase += 2 * math.pi * f / fps
        pulse = math.sin(phase) + 0.25 * math.sin(2 * phase)  # dicrotic-ish shape
        t_ms.append(int(t * 1000))
        red.append(round(185 + 2.2 * pulse + rng.gauss(0, 0.35) + 0.01 * t, 3))
        green.append(round(62 + 1.6 * pulse + rng.gauss(0, 0.35), 3))
        blue.append(round(21 + rng.gauss(0, 0.2), 3))
    return PPGRecording(started_at=start, fps=fps, t_ms=t_ms, red=red, green=green, blue=blue)


def synth_batch(now: datetime, tz: ZoneInfo, rng: random.Random) -> SensorBatch:
    local_now = now.astimezone(tz)
    # --- last night's sleep: screen off 23:30 -> unlock 06:45 local
    wake = local_now.replace(hour=6, minute=45, second=0, microsecond=0)
    if wake > local_now - timedelta(minutes=30):
        wake -= timedelta(days=1)
    bed = (wake - timedelta(days=1)).replace(hour=23, minute=30)
    screen = [
        ScreenEvent(at=bed - timedelta(minutes=40), type="unlock"),
        ScreenEvent(at=bed, type="screen_off"),
        ScreenEvent(at=wake, type="screen_on"),
        ScreenEvent(at=wake + timedelta(seconds=5), type="unlock"),
    ]
    t = wake + timedelta(minutes=20)
    while t < now - timedelta(minutes=5):  # daytime use: unlock, then screen off after 3–15 min
        screen.append(ScreenEvent(at=t, type="unlock"))
        screen.append(ScreenEvent(at=t + timedelta(minutes=rng.randint(3, 15)), type="screen_off"))
        t += timedelta(minutes=rng.randint(25, 70))
    screen = [e for e in screen if now - timedelta(hours=24) <= e.at <= now]

    # --- app sessions in the last 6 h
    sessions, t = [], now - timedelta(hours=6)
    while t < now - timedelta(minutes=10):
        pkg, cat = rng.choice(APPS)
        dur = timedelta(minutes=rng.randint(1, 18))
        sessions.append(AppUsageSession(start=t, end=t + dur, category=cat, app_hash=app_hash(pkg)))
        t += dur + timedelta(minutes=rng.randint(2, 25))

    # --- keystroke timings: three typing bursts in the last 50 min (no characters, only key classes)
    keys = []
    for burst_start in (now - timedelta(minutes=50), now - timedelta(minutes=30), now - timedelta(minutes=8)):
        t = burst_start
        for _ in range(rng.randint(70, 110)):
            t += timedelta(milliseconds=max(60, rng.gauss(210, 70)))
            r = rng.random()
            cls = "backspace" if r < 0.08 else "space" if r < 0.25 else "letter"
            keys.append(KeystrokeEvent(down_at=t, hold_ms=int(max(30, rng.gauss(95, 20))), key_class=cls))

    # --- activity: a 25-minute walk this morning
    walk = wake + timedelta(hours=1, minutes=15)
    activity = [
        ActivityTransition(at=walk, activity="walking", entering=True),
        ActivityTransition(at=walk + timedelta(minutes=25), activity="walking", entering=False),
        ActivityTransition(at=walk + timedelta(minutes=25), activity="still", entering=True),
    ]
    activity = [a for a in activity if a.at <= now]

    # --- hourly step counts and ambient light for the last 24 h
    steps, light = [], []
    for h in range(24, 0, -1):
        s = now - timedelta(hours=h)
        lh = s.astimezone(tz).hour
        awake = 7 <= lh <= 23
        steps.append(
            StepCount(
                start=s, end=s + timedelta(hours=1), steps=(rng.randint(150, 900) if awake else rng.randint(0, 15))
            )
        )
        light.append(
            LightReading(
                at=s,
                lux=round(
                    rng.uniform(80, 450) if 7 <= lh <= 18 else rng.uniform(30, 120) if awake else rng.uniform(0, 2), 1
                ),
            )
        )
    if walk <= now:
        steps.append(StepCount(start=walk, end=walk + timedelta(minutes=25), steps=2800))

    return SensorBatch(
        batch_id=uuid.uuid4(),
        device_id="sim-pixel-7",
        app_version="0.1.0-sim",
        timezone=str(tz),
        collected_from=now - timedelta(hours=24),
        collected_to=now,
        ppg=[synth_ppg(now - timedelta(minutes=20), rng)],
        keystrokes=keys,
        screen_events=screen,
        app_usage=sessions,
        activity=activity,
        steps=steps,
        light=light,
        self_reports=[
            SelfReport(at=now - timedelta(minutes=15), mood=3, stress=4, energy=2, focus=3, note="deadline week")
        ],
    )


def show(title: str):
    print(f"\n{'=' * 8} {title} {'=' * (60 - len(title))}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Simulate the phone app against the core API")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--timezone", default="Asia/Colombo")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    tz, rng = ZoneInfo(a.timezone), random.Random(a.seed)
    now = datetime.now(UTC).replace(microsecond=0)

    c = client(a.base_url)
    login_or_register(c, tz=a.timezone)
    tasks = {t["id"]: t["title"] for t in ensure_demo_tasks(c, now.astimezone(tz))}
    show("tasks")
    for title in tasks.values():
        print("  -", title)

    batch = synth_batch(now, tz, rng)
    r = c.post("/v1/sensing/batches", json=batch.model_dump(mode="json"))
    if r.status_code not in (200, 202):
        fail("sensor upload rejected", r)
    show("sensor upload")
    print(f"  batch {batch.batch_id} -> {r.json()}")

    r = c.post("/v1/pipeline/run")
    if r.status_code not in (200, 201, 202):
        fail("pipeline run failed", r)
    run = r.json()
    show(f"pipeline: {run.get('status')}")
    for s in run.get("steps", []):
        flag = " (fallback)" if s.get("used_fallback") else ""
        print(f"  {s['step']:<16} {'ok ' if s['ok'] else 'ERR'} {s['ms']:>5} ms{flag} {s.get('error') or ''}")

    ins = c.get("/v1/insights/latest").json()
    show("affective state (M1)")
    aff = ins.get("affective")
    if aff:
        f = aff.get("features", {})
        print(f"  {aff['label']} (confidence {aff['confidence']:.2f})")
        print(
            f"  heart rate {f.get('hr_bpm')} bpm · HRV RMSSD {f.get('hrv_rmssd_ms')} ms · "
            f"typing {f.get('typing_speed_kpm')} keys/min · backspace {f.get('backspace_rate')}"
        )
    show("wellness forecast (M2, next 12 slots)")
    wel = ins.get("wellness")
    if wel:
        print(f"  sleep last night: {wel.get('sleep_hours_last_night')} h")
        for s in wel["slots"][:12]:
            st = datetime.fromisoformat(s["start"]).astimezone(tz).strftime("%a %H:%M")
            print(
                f"  {st}  energy {s['energy']:.2f}  focus {s['focus']:.2f}  "
                f"fatigue {s['mental_fatigue']:.2f}  burnout {s['burnout_risk']:.2f}"
            )

    show("schedule (M3)")
    r = c.get("/v1/schedule/current")
    if r.status_code == 200:
        sch = r.json()
        for it in sch["items"]:
            st = datetime.fromisoformat(it["start"]).astimezone(tz)
            en = datetime.fromisoformat(it["end"]).astimezone(tz)
            fixed = " [fixed]" if it.get("is_fixed") else ""
            print(
                f"  {st:%a %H:%M}-{en:%H:%M}  p={it['completion_probability']:.2f}  "
                f"{tasks.get(it['task_id'], it['task_id'])}{fixed}"
            )
        for u in sch.get("unscheduled", []):
            print(f"  UNSCHEDULED {tasks.get(u['task_id'], u['task_id'])}: {u['reason']}")
        print(f"  overload predicted: {sch.get('overload_predicted')}  at risk: {len(sch.get('at_risk_task_ids', []))}")
    else:
        print(f"  no schedule ({r.status_code})")

    show("meal recommendations (M4)")
    r = c.get("/v1/recommendations/latest")
    if r.status_code == 200:
        rec = r.json()
        for m in rec["meals"]:
            st = datetime.fromisoformat(m["window_start"]).astimezone(tz)
            print(f"  {st:%a %H:%M} {m['meal_type']:<9} " + ", ".join(f["name"] for f in m["foods"]))
        for s in rec.get("suggested_activities", []):
            print(f"  suggestion: {s['title']} — {s['reason']}")
    else:
        print(f"  no recommendations ({r.status_code})")


if __name__ == "__main__":
    main()
