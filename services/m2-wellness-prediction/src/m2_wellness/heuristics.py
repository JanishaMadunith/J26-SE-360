"""Module 2 baseline: transparent, rule-based wellness forecast (two-process model of alertness).

Components (all scores clipped to 0–1):
* Sleep (last night)  — longest screen-off gap overlapping 21:00–11:00 local; minutes with recorded
                        steps inside the gap are removed; a bright gap (mean lux > 30) counts 80 %.
* Process C (clock)   — circadian alertness: 24 h cosine peaking ~16:00 plus a 12 h term giving the
                        post-lunch dip (~14:30) and a late-morning high.
* Process S (pressure)— homeostatic fatigue 1 − exp(−hours_awake / 18), plus sleep debt vs 7.5 h.
* Context             — workload due in 24 h, overdue tasks, recent stress/anxiety share from Module 1,
                        and the latest self-reported energy (decays over ~6 h).

These are starting assumptions to be replaced by the trained time-series model, not findings.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from wellness_contracts import (
    AffectiveLabel,
    AffectiveStateResult,
    LightReading,
    ScreenEvent,
    ScreenEventType,
    SelfReport,
    StepCount,
    WorkloadSummary,
)

NIGHT_START_H, NIGHT_END_H = 21, 11  # a sleep gap must overlap this local window
DEFAULT_SLEEP_H, DEFAULT_WAKE_H = 7.0, 7.0
SLEEP_NEED_H = 7.5


def clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def estimate_sleep(
    events: Sequence[ScreenEvent],
    steps: Sequence[StepCount],
    light: Sequence[LightReading],
    as_of: datetime,
    tz: ZoneInfo,
) -> tuple[float | None, datetime | None]:
    """Returns (sleep_hours, wake_time) for the most recent night, or (None, None) if unknown."""
    ev = sorted((e for e in events if e.at <= as_of and e.at >= as_of - timedelta(hours=36)), key=lambda e: e.at)
    gaps: list[tuple[datetime, datetime]] = []
    off_at = None
    for e in ev:
        if e.type == ScreenEventType.screen_off:
            off_at = off_at or e.at
        elif off_at is not None:  # screen_on or unlock ends the gap
            gaps.append((off_at, e.at))
            off_at = None
    best, best_len = None, 0.0
    for start, end in gaps:
        ls, le = start.astimezone(tz), end.astimezone(tz)
        overlaps_night = (
            ls.hour >= NIGHT_START_H
            or ls.hour < NIGHT_END_H
            or le.hour < NIGHT_END_H
            or (le - ls) > timedelta(hours=10)
        )
        length = (end - start).total_seconds() / 3600
        if overlaps_night and 2.0 <= length <= 16.0 and length > best_len:
            best, best_len = (start, end), length
    if best is None:
        return None, None
    start, end = best
    moving = (
        sum(
            min(s.end, end).timestamp() - max(s.start, start).timestamp()
            for s in steps
            if s.steps > 20 and s.end > start and s.start < end
        )
        / 3600
    )
    hours = max(0.0, best_len - max(0.0, moving))
    lux = [x.lux for x in light if start <= x.at <= end]
    if lux and sum(lux) / len(lux) > 30:
        hours *= 0.8
    return round(min(hours, 12.0), 2), end


def circadian(clock_h: float) -> float:
    """Process C: alertness by local clock time (0–1)."""
    return clip01(
        0.5 + 0.3 * math.cos(2 * math.pi * (clock_h - 16) / 24) + 0.07 * math.cos(2 * math.pi * (clock_h - 8.5) / 12)
    )


def homeostatic_fatigue(hours_awake: float) -> float:
    """Process S: rises with time awake (≈0.59 after 16 h)."""
    return 1 - math.exp(-max(0.0, hours_awake) / 18)


def stress_share(history: Sequence[AffectiveStateResult], as_of: datetime, hours: float) -> float:
    recent = [h for h in history if h.window_end >= as_of - timedelta(hours=hours)]
    if not recent:
        return 0.0
    w = sum(h.confidence for h in recent) or 1.0
    s = sum(h.confidence for h in recent if h.label in (AffectiveLabel.stress, AffectiveLabel.anxiety))
    return s / w


def latest_energy_report(reports: Sequence[SelfReport], as_of: datetime) -> tuple[float, datetime] | None:
    rs = [r for r in reports if r.energy is not None and as_of - timedelta(hours=12) <= r.at <= as_of]
    if not rs:
        return None
    r = max(rs, key=lambda x: x.at)
    return (r.energy - 1) / 4, r.at


def workload_load(w: WorkloadSummary) -> tuple[float, float]:
    return clip01(w.minutes_due_24h / 480), clip01(w.tasks_overdue / 5)
