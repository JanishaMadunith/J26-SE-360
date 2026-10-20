from datetime import timedelta
from zoneinfo import ZoneInfo

import numpy as np

from m2_wellness.heuristics import estimate_sleep, stress_share

FEATURES = [
    "sleep_h",
    "wake_hour",
    "stress_last",
    "stress_mean7",
    "stress_trend",
    "aff_conf",
    "steps_24h",
    "tasks_due_24h",
    "minutes_due_24h",
    "tasks_overdue",
    "tasks_pending",
    "asof_hour",
    "offset_h",
    "hour_sin",
    "hour_cos",
    "hours_awake",
]
TARGETS = ["mental_fatigue", "focus", "energy", "burnout_risk"]


def steps_last_24h(req) -> float:
    window_start = req.as_of - timedelta(hours=24)
    total = 0.0
    for reading in req.steps:
        overlap_start = max(reading.start, window_start)
        overlap_end = min(reading.end, req.as_of)
        duration = (reading.end - reading.start).total_seconds()
        overlap = (overlap_end - overlap_start).total_seconds()
        if duration > 0 and overlap > 0:
            total += reading.steps * overlap / duration
    return total


def slot_features(hour: float, wake_hour: float, offset_h: float) -> dict:
    """Build time-derived features for one forecast slot."""
    angle = 2 * np.pi * hour / 24
    hours_awake = (hour - wake_hour) % 24
    return {
        "offset_h": offset_h,
        "hour_sin": np.sin(angle),
        "hour_cos": np.cos(angle),
        "hours_awake": hours_awake,
    }


def request_features(req) -> dict:
    tz = ZoneInfo(req.timezone)
    local = req.as_of.astimezone(tz)

    sleep_h, wake = estimate_sleep(req.screen_events, req.steps, req.light, req.as_of, tz)
    wake_hour = np.nan
    if wake is not None:
        w = wake.astimezone(tz)
        wake_hour = w.hour + w.minute / 60

    hist = req.affective_history
    if hist:
        s24 = stress_share(hist, req.as_of, 24)
        s7 = stress_share(hist, req.as_of, 24 * 7)
        stress_last, stress_mean7, stress_trend = s24, s7, s24 - s7
        aff_conf = float(np.mean([r.confidence for r in hist]))
    else:  # stress_share may return 0 on empty history, which would wrongly mean "calm"
        stress_last = stress_mean7 = stress_trend = aff_conf = np.nan

    w_ = req.workload
    return {
        "sleep_h": np.nan if sleep_h is None else float(sleep_h),
        "wake_hour": wake_hour,
        "stress_last": stress_last,
        "stress_mean7": stress_mean7,
        "stress_trend": stress_trend,
        "aff_conf": aff_conf,
        "steps_24h": steps_last_24h(req),
        "tasks_due_24h": w_.tasks_due_24h,
        "minutes_due_24h": w_.minutes_due_24h,
        "tasks_overdue": w_.tasks_overdue,
        "tasks_pending": w_.tasks_pending,
        "asof_hour": local.hour + local.minute / 60,
    }
