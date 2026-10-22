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
    "sleep_mean3",
    "sleep_trend3",
    "steps_mean3",
    "steps_trend3",
    "stress_mean3",
    "fatigue_mean3",
    "fatigue_trend3",
    "focus_mean3",
    "focus_trend3",
    "energy_mean3",
    "energy_trend3",
    "overwhelmed_mean3",
    "overwhelmed_trend3",
    "is_sleeping",
    "hours_until_bedtime",
    "planned_sleep_duration",
    "sleep_debt",
    "current_fatigue",
    "current_focus",
    "current_energy",
    "current_overwhelmed",
    "previous_fatigue",
    "previous_focus",
    "previous_energy",
    "previous_burnout",
    "bedtime_hour",
    "next_wake_hour",
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


def slot_features(
    hour: float,
    wake_hour: float,
    offset_h: float,
    bedtime_hour: float = 23.0,
    next_wake_hour: float = 7.0,
    asof_hour: float = 9.0,
) -> dict:
    """Build time-derived features for one forecast slot."""
    angle = 2 * np.pi * hour / 24
    absolute_hour = asof_hour + offset_h
    bedtime = np.floor(absolute_hour / 24) * 24 + bedtime_hour
    sleep_duration = (next_wake_hour - bedtime_hour) % 24
    previous_bedtime = bedtime - 24
    if absolute_hour < bedtime and absolute_hour < previous_bedtime + sleep_duration:
        bedtime = previous_bedtime
    wake_after_bedtime = bedtime + sleep_duration
    sleeping = bedtime <= absolute_hour < wake_after_bedtime
    if sleeping:
        hours_awake = 0.0
        hours_until_bedtime = 0.0
    else:
        wake = np.floor(absolute_hour / 24) * 24 + wake_hour
        if wake > absolute_hour:
            wake -= 24
        hours_awake = max(0.0, absolute_hour - wake)
        hours_until_bedtime = max(0.0, bedtime - absolute_hour)
    return {
        "offset_h": offset_h,
        "hour_sin": np.sin(angle),
        "hour_cos": np.cos(angle),
        "hours_awake": hours_awake,
        "is_sleeping": float(sleeping),
        "hours_until_bedtime": hours_until_bedtime,
        "planned_sleep_duration": sleep_duration,
    }


def manual_request_features(req) -> dict:
    manual = req.manual_inputs
    current = manual.current
    records = sorted([*manual.history, current], key=lambda record: record.date)
    stress_values = [record.stress for record in records]
    sleep_values = [record.sleep_hours for record in records]
    steps_values = [record.steps for record in records]
    fatigue_values = [record.mental_tiredness for record in records]
    focus_values = [record.focus for record in records]
    energy_values = [record.energy for record in records]
    overwhelmed_values = [record.overwhelmed for record in records]
    stress_mean = float(np.mean(stress_values))
    latest = records[-1]
    def trend(values):
        return float(values[-1] - values[0])
    stress_last = float(current.stress)
    tz = ZoneInfo(req.timezone)
    local = req.as_of.astimezone(tz)
    return {
        "sleep_h": float(current.sleep_hours),
        "sleep_debt": max(0.0, 7.5 - float(current.sleep_hours)),
        "wake_hour": float(current.wake_hour),
        "stress_last": stress_last,
        "stress_mean7": stress_mean,
        "stress_trend": stress_last - stress_mean,
        "aff_conf": 1.0,
        "steps_24h": float(latest.steps),
        "tasks_due_24h": current.tasks_due_24h,
        "minutes_due_24h": current.minutes_due_24h,
        "tasks_overdue": current.tasks_overdue,
        "tasks_pending": current.tasks_pending,
        "asof_hour": local.hour + local.minute / 60,
        "sleep_mean3": float(np.mean(sleep_values)),
        "sleep_trend3": trend(sleep_values),
        "steps_mean3": float(np.mean(steps_values)),
        "steps_trend3": trend(steps_values),
        "stress_mean3": stress_mean,
        "fatigue_mean3": float(np.mean(fatigue_values)),
        "fatigue_trend3": trend(fatigue_values),
        "focus_mean3": float(np.mean(focus_values)),
        "focus_trend3": trend(focus_values),
        "energy_mean3": float(np.mean(energy_values)),
        "energy_trend3": trend(energy_values),
        "overwhelmed_mean3": float(np.mean(overwhelmed_values)),
        "overwhelmed_trend3": trend(overwhelmed_values),
        "current_fatigue": float(latest.mental_tiredness),
        "current_focus": float(latest.focus),
        "current_energy": float(latest.energy),
        "current_overwhelmed": float(latest.overwhelmed),
        "previous_fatigue": np.nan,
        "previous_focus": np.nan,
        "previous_energy": np.nan,
        "previous_burnout": np.nan,
        "bedtime_hour": float(current.bedtime_hour),
        "next_wake_hour": float(current.next_wake_hour),
    }


def request_features(req) -> dict:
    if req.manual_inputs is not None:
        return manual_request_features(req)

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
        "sleep_mean3": np.nan,
        "sleep_debt": np.nan if sleep_h is None else max(0.0, 7.5 - float(sleep_h)),
        "sleep_trend3": np.nan,
        "steps_mean3": np.nan,
        "steps_trend3": np.nan,
        "stress_mean3": np.nan,
        "fatigue_mean3": np.nan,
        "fatigue_trend3": np.nan,
        "focus_mean3": np.nan,
        "focus_trend3": np.nan,
        "energy_mean3": np.nan,
        "energy_trend3": np.nan,
        "overwhelmed_mean3": np.nan,
        "overwhelmed_trend3": np.nan,
        "current_fatigue": np.nan,
        "current_focus": np.nan,
        "current_energy": np.nan,
        "current_overwhelmed": np.nan,
        "previous_fatigue": np.nan,
        "previous_focus": np.nan,
        "previous_energy": np.nan,
        "previous_burnout": np.nan,
        "bedtime_hour": 23.0,
        "next_wake_hour": 7.0,
    }
