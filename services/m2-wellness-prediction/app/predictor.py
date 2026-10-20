"""Module 2 predictor.

BASELINE (v0.1.0): two-process alertness model + context rules (see m2_wellness/heuristics.py) that
forecasts mental fatigue, focus, energy and burnout risk for every slot of the horizon. Slot-level output
is what Module 3 needs (resolves the daily-vs-intraday granularity issue).

REPLACE with the trained time-series model: keep the class name, `info` and `predict()` signature.
"""

from __future__ import annotations

import math
import pathlib
from datetime import timedelta
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

from m2_wellness.features import FEATURES, TARGETS, request_features, slot_features
from m2_wellness.heuristics import (
    DEFAULT_WAKE_H,
    SLEEP_NEED_H,
    circadian,
    clip01,
    estimate_sleep,
    homeostatic_fatigue,
    latest_energy_report,
    stress_share,
    workload_load,
)
from wellness_contracts import ModelInfo, WellnessForecast, WellnessForecastRequest, WellnessSlot, utc_now

MODELS = pathlib.Path(__file__).resolve().parents[1] / "models"


class Predictor:
    info = ModelInfo(name="m2-hgb-slot-forecaster", version="0.2.0")
    baseline_info = ModelInfo(name="m2-two-process-baseline", version="0.1.0")

    def __init__(self):
        files = [MODELS / f"{t}.joblib" for t in TARGETS]
        self.models = (
            {t: joblib.load(f) for t, f in zip(TARGETS, files, strict=False)}
            if all(f.exists() for f in files)
            else None
        )

    def predict(self, req: WellnessForecastRequest) -> WellnessForecast:
        if self.models is None:
            return self._baseline(req)

        tz = ZoneInfo(req.timezone)
        now = req.as_of.astimezone(tz)
        rf = request_features(req)
        if np.isnan(rf["wake_hour"]):
            rf["wake_hour"] = DEFAULT_WAKE_H
        sleep_h = None if np.isnan(rf["sleep_h"]) else float(rf["sleep_h"])

        step = timedelta(minutes=req.slot_minutes)
        local = now.replace(second=0, microsecond=0)
        start = local - timedelta(minutes=(local.hour * 60 + local.minute) % req.slot_minutes)
        n = req.horizon_hours * 60 // req.slot_minutes
        times = [start + i * step for i in range(n)]

        rows = []
        for t in times:
            sf = slot_features(t.hour + t.minute / 60, rf["wake_hour"], 0.0)
            sf["offset_h"] = max(0.0, (t - now).total_seconds() / 3600)
            rows.append({**rf, **sf})
        x = pd.DataFrame(rows)[FEATURES].astype(float)
        sc = {t: np.clip(self.models[t].predict(x), 0, 1) for t in TARGETS}

        slots = [
            WellnessSlot(
                start=t,
                end=t + step,
                mental_fatigue=round(float(sc["mental_fatigue"][i]), 4),
                focus=round(float(sc["focus"][i]), 4),
                energy=round(float(sc["energy"][i]), 4),
                burnout_risk=round(float(sc["burnout_risk"][i]), 4),
            )
            for i, t in enumerate(times)
        ]
        return WellnessForecast(
            user_id=req.user_id,
            generated_at=utc_now(),
            slot_minutes=req.slot_minutes,
            slots=slots,
            sleep_hours_last_night=sleep_h,
            model=self.info,
        )

    def _baseline(self, req: WellnessForecastRequest) -> WellnessForecast:
        tz = ZoneInfo(req.timezone)
        sleep_h, wake = estimate_sleep(req.screen_events, req.steps, req.light, req.as_of, tz)
        if wake is None:
            local = req.as_of.astimezone(tz)
            wake = local.replace(hour=int(DEFAULT_WAKE_H), minute=0, second=0, microsecond=0)
            if wake > local:
                wake -= timedelta(days=1)
        debt = clip01((SLEEP_NEED_H - (sleep_h if sleep_h is not None else 7.0)) / SLEEP_NEED_H)
        wl, overdue = workload_load(req.workload)
        stress_24h = stress_share(req.affective_history, req.as_of, 24)
        stress_7d = stress_share(req.affective_history, req.as_of, 24 * 7)
        report = latest_energy_report(req.self_reports, req.as_of)

        step = timedelta(minutes=req.slot_minutes)
        local = req.as_of.astimezone(tz).replace(second=0, microsecond=0)
        start = local - timedelta(minutes=(local.hour * 60 + local.minute) % req.slot_minutes)
        slots = []
        t = start
        end = start + timedelta(hours=req.horizon_hours)
        while t < end:
            mid = (t + step / 2).astimezone(tz)
            clock = mid.hour + mid.minute / 60
            awake_h = ((mid - wake.astimezone(tz)).total_seconds() / 3600) % 24
            c = circadian(clock)
            s = homeostatic_fatigue(awake_h)
            energy = clip01(0.2 + 0.8 * c - 0.35 * s - 0.25 * debt - 0.1 * stress_24h)
            if report is not None:
                val, at = report
                w = math.exp(-max(0.0, (mid - at).total_seconds() / 3600) / 6) * 0.6
                energy = clip01((1 - w) * energy + w * val)
            fatigue = clip01(0.15 + 0.55 * s + 0.25 * debt + 0.15 * wl + 0.15 * stress_24h - 0.1 * (c - 0.5))
            focus = clip01(0.6 * energy + 0.4 * (1 - fatigue) - 0.15 * stress_24h - 0.05 * wl)
            burnout = clip01(0.1 + 0.45 * stress_7d + 0.2 * overdue + 0.15 * wl + 0.15 * debt + 0.1 * fatigue)
            slots.append(
                WellnessSlot(
                    start=t,
                    end=t + step,
                    mental_fatigue=round(fatigue, 4),
                    focus=round(focus, 4),
                    energy=round(energy, 4),
                    burnout_risk=round(burnout, 4),
                )
            )
            t += step
        return WellnessForecast(
            user_id=req.user_id,
            generated_at=utc_now(),
            slot_minutes=req.slot_minutes,
            slots=slots,
            sleep_hours_last_night=sleep_h,
            model=self.baseline_info,
        )
