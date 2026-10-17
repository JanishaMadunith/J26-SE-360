"""Server-side feature extraction from RAW phone signals (Module 1).

Every function is pure and tolerant of missing data: if a signal is absent or too short, the
corresponding features are None (never invented). These same functions should be reused when building
training datasets, so serving and training compute features identically.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import numpy as np
from scipy import signal as sps

from wellness_contracts import (
    ActivityTransition,
    ActivityType,
    AppCategory,
    AppUsageSession,
    KeyClass,
    KeystrokeEvent,
    LightReading,
    PPGRecording,
    ScreenEvent,
    ScreenEventType,
    StepCount,
)

HR_BAND_HZ = (0.7, 3.5)  # 42–210 bpm
MIN_PPG_SECONDS = 10.0
TYPING_PAUSE_S = 5.0  # inter-key gaps longer than this are pauses, not typing rhythm
ACTIVE = {ActivityType.walking, ActivityType.running, ActivityType.on_bicycle}


# ------------------------------------------------------------------------------------------ PPG
def _ppg_one(rec: PPGRecording) -> dict | None:
    t = np.asarray(rec.t_ms, dtype=float) / 1000.0
    if len(t) < 3 or t[-1] - t[0] < MIN_PPG_SECONDS:
        return None
    fs = float(rec.fps)
    grid = np.arange(t[0], t[-1], 1.0 / fs)
    if len(grid) < int(MIN_PPG_SECONDS * fs):
        return None
    b, a = sps.butter(3, [HR_BAND_HZ[0] / (fs / 2), HR_BAND_HZ[1] / (fs / 2)], btype="band")
    best = None
    for name in ("red", "green"):
        x = np.interp(grid, t, np.asarray(getattr(rec, name), dtype=float))
        x = sps.detrend(x)
        if np.allclose(x.std(), 0):
            continue
        y = sps.filtfilt(b, a, x)
        nper = min(len(y), int(fs * 8))
        f, pxx = sps.welch(y, fs=fs, nperseg=nper, nfft=max(4096, nper))  # zero-pad: ~0.4 bpm bins
        band = (f >= HR_BAND_HZ[0]) & (f <= HR_BAND_HZ[1])
        if not band.any():
            continue
        quality = float(pxx[band].sum() / max(pxx.sum(), 1e-12))
        if best is None or quality > best["quality"]:
            best = {"y": y, "f": f[band], "p": pxx[band], "quality": quality, "channel": name}
    if best is None:
        return None
    hr_fft = float(best["f"][np.argmax(best["p"])] * 60.0)
    # beat detection for HRV: peaks at least 60/210 s apart, guided by the FFT heart rate
    min_dist = max(1, int(fs * 60.0 / min(210.0, hr_fft * 1.6)))
    peaks, _ = sps.find_peaks(best["y"], distance=min_dist, prominence=np.std(best["y"]) * 0.3)
    rmssd = None
    hr_peaks = None
    if len(peaks) >= 4:
        ibi = np.diff(grid[peaks])
        med = np.median(ibi)
        ibi = ibi[(ibi > 0.28) & (ibi < 1.5) & (np.abs(ibi - med) < 0.3 * med)]
        if len(ibi) >= 3:
            rmssd = float(np.sqrt(np.mean(np.diff(ibi * 1000.0) ** 2)))
            hr_peaks = float(60.0 / np.mean(ibi))
    # beat-based HR is more precise when it agrees with the spectral estimate
    hr = hr_peaks if hr_peaks is not None and abs(hr_peaks - hr_fft) / hr_fft < 0.1 else hr_fft
    return {
        "hr_bpm": hr,
        "hr_bpm_peaks": hr_peaks,
        "hrv_rmssd_ms": rmssd,
        "ppg_quality": best["quality"],
        "seconds": float(grid[-1] - grid[0]),
    }


def ppg_features(recordings: Sequence[PPGRecording]) -> dict[str, float | None]:
    """Heart rate (Welch PSD peak in 0.7–3.5 Hz) and HRV RMSSD (from detected beats), averaged over
    recordings weighted by duration. The channel (red/green) with the stronger pulse band is used."""
    out = [r for r in (_ppg_one(rec) for rec in recordings) if r is not None]
    if not out:
        return {"hr_bpm": None, "hrv_rmssd_ms": None, "ppg_quality": None, "ppg_seconds": 0.0}
    w = np.array([o["seconds"] for o in out])

    def wavg(key):
        vals = [(o[key], wi) for o, wi in zip(out, w, strict=True) if o[key] is not None]
        if not vals:
            return None
        v, ww = zip(*vals, strict=True)
        return float(np.average(v, weights=ww))

    return {
        "hr_bpm": wavg("hr_bpm"),
        "hrv_rmssd_ms": wavg("hrv_rmssd_ms"),
        "ppg_quality": wavg("ppg_quality"),
        "ppg_seconds": float(w.sum()),
    }


# ------------------------------------------------------------------------------------------ typing
def typing_features(keys: Sequence[KeystrokeEvent]) -> dict[str, float | None]:
    if len(keys) < 5:
        return {
            "typing_keys": float(len(keys)),
            "typing_speed_kpm": None,
            "iki_mean_ms": None,
            "iki_std_ms": None,
            "backspace_rate": None,
            "hold_mean_ms": None,
        }
    ks = sorted(keys, key=lambda k: k.down_at)
    t = np.array([k.down_at.timestamp() for k in ks])
    iki = np.diff(t)
    rhythm = iki[(iki > 0) & (iki <= TYPING_PAUSE_S)]
    active_min = rhythm.sum() / 60.0
    holds = np.array([k.hold_ms for k in ks if k.hold_ms > 0], dtype=float)
    return {
        "typing_keys": float(len(ks)),
        "typing_speed_kpm": float(len(rhythm) / active_min) if active_min > 0 else None,
        "iki_mean_ms": float(rhythm.mean() * 1000) if len(rhythm) else None,
        "iki_std_ms": float(rhythm.std() * 1000) if len(rhythm) > 1 else None,
        "backspace_rate": float(sum(k.key_class == KeyClass.backspace for k in ks) / len(ks)),
        "hold_mean_ms": float(holds.mean()) if len(holds) else None,
    }


# ------------------------------------------------------------------------------------------ phone use
def screen_features(events: Sequence[ScreenEvent], hours: float) -> dict[str, float | None]:
    ev = sorted(events, key=lambda e: e.at)
    unlocks = sum(e.type == ScreenEventType.unlock for e in ev)
    on_s, on_at = 0.0, None
    for e in ev:
        if e.type in (ScreenEventType.screen_on, ScreenEventType.unlock) and on_at is None:
            on_at = e.at
        elif e.type == ScreenEventType.screen_off and on_at is not None:
            on_s += (e.at - on_at).total_seconds()
            on_at = None
    return {"unlocks_per_hour": unlocks / hours if ev else None, "screen_on_minutes": on_s / 60.0 if ev else None}


def app_features(sessions: Sequence[AppUsageSession], hours: float) -> dict[str, float | None]:
    if not sessions:
        return {"app_switches_per_hour": None, "social_share": None, "distinct_apps": None}
    total = sum(max(0.0, (s.end - s.start).total_seconds()) for s in sessions) or 1.0
    social = sum(
        max(0.0, (s.end - s.start).total_seconds())
        for s in sessions
        if s.category in (AppCategory.social, AppCategory.video, AppCategory.game)
    )
    return {
        "app_switches_per_hour": len(sessions) / hours,
        "social_share": social / total,
        "distinct_apps": float(len({s.app_hash for s in sessions})),
    }


def activity_features(
    transitions: Sequence[ActivityTransition], steps: Sequence[StepCount], window_start: datetime, window_end: datetime
) -> dict[str, float | None]:
    hours = max((window_end - window_start).total_seconds() / 3600.0, 0.25)
    active_s = 0.0
    started: dict[ActivityType, datetime] = {}
    for tr in sorted(transitions, key=lambda x: x.at):
        if tr.activity not in ACTIVE:
            continue
        if tr.entering:
            started[tr.activity] = tr.at
        elif tr.activity in started:
            active_s += (tr.at - started.pop(tr.activity)).total_seconds()
    for _act, at in started.items():  # still active at window end
        active_s += max(0.0, (window_end - at).total_seconds())
    total_steps = sum(s.steps for s in steps)
    return {
        "active_minutes": active_s / 60.0 if transitions else None,
        "steps_per_hour": total_steps / hours if steps else None,
    }


def light_features(light: Sequence[LightReading]) -> dict[str, float | None]:
    return {"light_mean_lux": float(np.mean([x.lux for x in light])) if light else None}


def extract_all(req) -> dict[str, float | None]:
    """All Module 1 features for one AffectiveStateRequest."""
    hours = max((req.window_end - req.window_start).total_seconds() / 3600.0, 0.25)
    f: dict[str, float | None] = {}
    f.update(ppg_features(req.ppg))
    f.update(typing_features(req.keystrokes))
    f.update(screen_features(req.screen_events, hours))
    f.update(app_features(req.app_usage, hours))
    f.update(activity_features(req.activity, req.steps, req.window_start, req.window_end))
    f.update(light_features(req.light))
    return f
