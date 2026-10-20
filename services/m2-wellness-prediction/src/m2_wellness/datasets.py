"""Process StudentLife multimodal dataset into M2 forecast training windows."""
from __future__ import annotations

import glob
import json
import math
import os
import pathlib
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from m2_wellness.features import FEATURES, TARGETS, slot_features
from m2_wellness.heuristics import DEFAULT_WAKE_H, circadian, clip01, homeostatic_fatigue

STUDENTLIFE_TZ = ZoneInfo("America/New_York")


def parse_surveys(survey_dir: pathlib.Path) -> dict[str, dict[str, float]]:
    """Parse PHQ-9 and PSS surveys to compute baseline burnout & stress scores per user."""
    user_scores: dict[str, dict[str, float]] = {}

    phq_path = survey_dir / "PHQ-9.csv"
    if phq_path.exists():
        phq = pd.read_csv(phq_path)
        phq_items = [c for c in phq.columns if c not in ["uid", "type", "Response"]]
        phq_map = {"Not at all": 0, "Several days": 1, "More than half the days": 2, "Nearly every day": 3}
        for uid, grp in phq.groupby("uid"):
            scores = []
            for _, row in grp.iterrows():
                val = sum(phq_map.get(row[col], 0) for col in phq_items) / 27.0
                scores.append(val)
            user_scores.setdefault(str(uid), {})["phq"] = float(np.mean(scores))

    pss_path = survey_dir / "PerceivedStressScale.csv"
    if pss_path.exists():
        pss = pd.read_csv(pss_path)
        pss_cols = [c for c in pss.columns if c not in ["uid", "type"]]
        pss_map = {"Never": 0, "Almost never": 1, "Sometime": 2, "Fairly often": 3, "Very often": 4}
        # In PSS-10, items 4, 5, 7, 8 are positive items (reverse scored)
        reverse_idx = {3, 4, 6, 7}
        for uid, grp in pss.groupby("uid"):
            scores = []
            for _, row in grp.iterrows():
                row_scores = []
                for idx, col in enumerate(pss_cols):
                    v = pss_map.get(row[col], 2)
                    if idx in reverse_idx:
                        v = 4 - v
                    row_scores.append(v)
                scores.append(sum(row_scores) / 40.0)
            user_scores.setdefault(str(uid), {})["pss"] = float(np.mean(scores))

    return user_scores


def parse_deadlines(edu_dir: pathlib.Path) -> dict[tuple[str, str], int]:
    """Parse deadlines.csv into a mapping of (uid, date_str) -> deadline count."""
    deadlines: dict[tuple[str, str], int] = {}
    path = edu_dir / "deadlines.csv"
    if not path.exists():
        return deadlines

    df = pd.read_csv(path).set_index("uid")
    for uid, row in df.iterrows():
        for d_str, count in row.items():
            if pd.notna(count):
                deadlines[(str(uid), str(d_str))] = int(count)
    return deadlines


def parse_sensing_steps(sensing_dir: pathlib.Path) -> dict[tuple[str, str], float]:
    """Parse sensing/activity CSV files to calculate daily step counts per user."""
    steps_map: dict[tuple[str, str], float] = {}
    act_files = glob.glob(str(sensing_dir / "activity" / "*.csv"))
    for f in act_files:
        uid = pathlib.Path(f).stem.replace("activity_", "")
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip()
        if "timestamp" not in df.columns or "activity inference" not in df.columns:
            continue
        active = df[df["activity inference"].isin([1, 2])]
        if active.empty:
            continue
        dates = pd.to_datetime(active["timestamp"], unit="s", utc=True).dt.tz_convert(STUDENTLIFE_TZ).dt.strftime("%Y-%m-%d")
        step_val = (active["activity inference"] == 1) * 4.5 + (active["activity inference"] == 2) * 8.0
        daily = step_val.groupby(dates).sum()
        for d, s in daily.items():
            steps_map[(uid, str(d))] = float(s)
    return steps_map


def parse_sensing_sleep(sensing_dir: pathlib.Path) -> dict[tuple[str, str], tuple[float, float]]:
    """Parse sensing dark/phonelock to find longest nocturnal sleep interval per user-day."""
    sleep_map: dict[tuple[str, str], tuple[float, float]] = {}
    dark_files = glob.glob(str(sensing_dir / "dark" / "*.csv"))
    for f in dark_files:
        uid = pathlib.Path(f).stem.replace("dark_", "")
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip()
        if "start" not in df.columns or "end" not in df.columns:
            continue
        for _, row in df.iterrows():
            st = datetime.fromtimestamp(row["start"], tz=STUDENTLIFE_TZ)
            en = datetime.fromtimestamp(row["end"], tz=STUDENTLIFE_TZ)
            dur = (en - st).total_seconds() / 3600.0
            if 3.0 <= dur <= 14.0 and (st.hour >= 20 or st.hour <= 5):
                wake_date = en.strftime("%Y-%m-%d")
                wake_hour = en.hour + en.minute / 60.0
                curr = sleep_map.get((uid, wake_date))
                if curr is None or dur > curr[0]:
                    sleep_map[(uid, wake_date)] = (round(dur, 2), round(wake_hour, 2))
    return sleep_map


def parse_ema_stress(ema_dir: pathlib.Path) -> dict[str, list[dict]]:
    """Parse Stress EMA records."""
    stress_map: dict[str, list[dict]] = {}
    level_scores = {"1": 0.40, "2": 0.70, "3": 0.95, "4": 0.20, "5": 0.05}
    files = glob.glob(str(ema_dir / "response" / "Stress" / "*.json"))
    for f in files:
        uid = pathlib.Path(f).stem.replace("Stress_", "")
        with open(f, encoding="utf-8") as fp:
            data = json.load(fp)
        user_list = []
        for r in data:
            if "resp_time" in r and "level" in r and str(r["level"]) in level_scores:
                dt = datetime.fromtimestamp(r["resp_time"], tz=STUDENTLIFE_TZ)
                user_list.append({"dt": dt, "score": level_scores[str(r["level"])]})
        user_list.sort(key=lambda x: x["dt"])
        stress_map[uid] = user_list
    return stress_map


def parse_ema_sleep(ema_dir: pathlib.Path) -> dict[tuple[str, str], dict]:
    """Parse Sleep EMA records."""
    sleep_ema: dict[tuple[str, str], dict] = {}
    hour_map = {
        "1": 2.5, "2": 3.5, "3": 4.0, "4": 4.5, "5": 5.0, "6": 5.5, "7": 6.0, "8": 6.5,
        "9": 7.0, "10": 7.5, "11": 8.0, "12": 8.5, "13": 9.0, "14": 9.5, "15": 10.0,
        "16": 10.5, "17": 11.0, "18": 11.5, "19": 12.0,
    }
    files = glob.glob(str(ema_dir / "response" / "Sleep" / "*.json"))
    for f in files:
        uid = pathlib.Path(f).stem.replace("Sleep_", "")
        with open(f, encoding="utf-8") as fp:
            data = json.load(fp)
        for r in data:
            if "resp_time" in r and "hour" in r and str(r["hour"]) in hour_map:
                dt = datetime.fromtimestamp(r["resp_time"], tz=STUDENTLIFE_TZ)
                d_str = dt.strftime("%Y-%m-%d")
                sleep_ema[(uid, d_str)] = {
                    "hours": hour_map[str(r["hour"])],
                    "rate": int(r.get("rate", 2)) if str(r.get("rate", "")).isdigit() else 2,
                    "social_trouble": int(r.get("social", 1)) if str(r.get("social", "")).isdigit() else 1,
                }
    return sleep_ema


def parse_ema_pam(ema_dir: pathlib.Path) -> dict[str, list[dict]]:
    """Parse PAM (Photographic Affect Meter) EMA records."""
    pam_map: dict[str, list[dict]] = {}
    files = glob.glob(str(ema_dir / "response" / "PAM" / "*.json"))
    for f in files:
        uid = pathlib.Path(f).stem.replace("PAM_", "")
        with open(f, encoding="utf-8") as fp:
            data = json.load(fp)
        records = []
        for r in data:
            if "resp_time" in r and "picture_idx" in r:
                idx = int(r["picture_idx"])
                if 1 <= idx <= 16:
                    dt = datetime.fromtimestamp(r["resp_time"], tz=STUDENTLIFE_TZ)
                    # 4x4 matrix: row is arousal (0 to 3 -> low to high)
                    arousal = ((idx - 1) // 4) / 3.0
                    valence = ((idx - 1) % 4) / 3.0
                    records.append({"dt": dt, "arousal": arousal, "valence": valence})
        records.sort(key=lambda x: x["dt"])
        pam_map[uid] = records
    return pam_map


def parse_ema_productivity(ema_dir: pathlib.Path) -> dict[tuple[str, str], float]:
    """Parse Study Spaces EMA productivity ratings."""
    prod_map: dict[tuple[str, str], float] = {}
    score_map = {"1": 0.15, "2": 0.40, "3": 0.70, "4": 0.95}
    files = glob.glob(str(ema_dir / "response" / "Study Spaces" / "*.json"))
    for f in files:
        uid = pathlib.Path(f).stem.replace("Study Spaces_", "")
        with open(f, encoding="utf-8") as fp:
            data = json.load(fp)
        for r in data:
            if "resp_time" in r and "productivity" in r and str(r["productivity"]) in score_map:
                dt = datetime.fromtimestamp(r["resp_time"], tz=STUDENTLIFE_TZ)
                prod_map[(uid, dt.strftime("%Y-%m-%d"))] = score_map[str(r["productivity"])]
    return prod_map


def build_studentlife_dataset(
    dataset_dir: str | pathlib.Path,
    output_csv: str | pathlib.Path | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Build unified multimodal training dataset from StudentLife files."""
    ds_path = pathlib.Path(dataset_dir)
    if (ds_path / "dataset").exists():
        ds_path = ds_path / "dataset"

    rng = np.random.default_rng(seed)

    print("Parsing surveys...")
    surveys = parse_surveys(ds_path / "survey")

    print("Parsing deadlines...")
    deadlines = parse_deadlines(ds_path / "education")

    print("Parsing steps from sensing activity...")
    steps_map = parse_sensing_steps(ds_path / "sensing")

    print("Parsing sleep from sensing dark...")
    sensing_sleep = parse_sensing_sleep(ds_path / "sensing")

    print("Parsing EMA responses (Stress, Sleep, PAM, Productivity)...")
    ema_stress = parse_ema_stress(ds_path / "EMA")
    ema_sleep = parse_ema_sleep(ds_path / "EMA")
    ema_pam = parse_ema_pam(ds_path / "EMA")
    ema_prod = parse_ema_productivity(ds_path / "EMA")

    # Collect all user IDs
    all_uids = sorted(
        set(ema_stress.keys())
        | {u for u, _ in sensing_sleep}
        | {u for u, _ in steps_map}
        | {u for u, _ in deadlines}
    )

    # Date range across study: March 28, 2013 to June 1, 2013
    start_date = datetime(2013, 3, 28).date()
    end_date = datetime(2013, 6, 1).date()
    days_span = (end_date - start_date).days

    rows = []
    print(f"Generating forecast training windows for {len(all_uids)} users...")

    for uid in all_uids:
        u_survey = surveys.get(uid, {})
        phq = u_survey.get("phq", 0.15)
        pss = u_survey.get("pss", 0.35)
        burnout_base = 0.5 * phq + 0.5 * pss

        u_stress_list = ema_stress.get(uid, [])
        u_pam_list = ema_pam.get(uid, [])

        for day_offset in range(days_span):
            curr_date = start_date + timedelta(days=day_offset)
            d_str = curr_date.strftime("%Y-%m-%d")

            # Check if this user was active around this date
            has_steps = (uid, d_str) in steps_map
            has_sleep = (uid, d_str) in ema_sleep or (uid, d_str) in sensing_sleep
            has_deadlines = (uid, d_str) in deadlines
            if not (has_steps or has_sleep or has_deadlines):
                continue

            # 1. Sleep & wake
            if (uid, d_str) in ema_sleep:
                sleep_info = ema_sleep[(uid, d_str)]
                sleep_h = float(sleep_info["hours"])
                wake_h = sensing_sleep.get((uid, d_str), (sleep_h, DEFAULT_WAKE_H))[1]
                social_trouble = sleep_info["social_trouble"]
            elif (uid, d_str) in sensing_sleep:
                dur, wake_h = sensing_sleep[(uid, d_str)]
                sleep_h = float(dur)
                social_trouble = 1
            else:
                sleep_h = np.nan
                wake_h = np.nan
                social_trouble = 1

            # 2. Steps in last 24h
            steps_24h = steps_map.get((uid, d_str), float(rng.normal(6000, 2000)))
            steps_24h = float(np.clip(steps_24h, 200, 25000))

            # 3. Workload from deadlines
            tasks_due_24h = deadlines.get((uid, d_str), 0)
            minutes_due_24h = tasks_due_24h * 120.0
            # overdue: sum of past 7 days
            tasks_overdue = sum(
                deadlines.get((uid, (curr_date - timedelta(days=i)).strftime("%Y-%m-%d")), 0)
                for i in range(1, 8)
            )
            tasks_pending = sum(
                deadlines.get((uid, (curr_date + timedelta(days=i)).strftime("%Y-%m-%d")), 0)
                for i in range(1, 8)
            )

            # 4. Stress history (as of query time ~09:00 local)
            asof_dt = datetime(curr_date.year, curr_date.month, curr_date.day, 9, 0, tzinfo=STUDENTLIFE_TZ)
            recent_stress_24h = [
                s["score"] for s in u_stress_list
                if asof_dt - timedelta(hours=24) <= s["dt"] <= asof_dt
            ]
            recent_stress_7d = [
                s["score"] for s in u_stress_list
                if asof_dt - timedelta(days=7) <= s["dt"] <= asof_dt
            ]

            if recent_stress_24h:
                stress_last = float(recent_stress_24h[-1])
            elif recent_stress_7d:
                stress_last = float(recent_stress_7d[-1])
            else:
                stress_last = np.nan

            stress_mean7 = float(np.mean(recent_stress_7d)) if recent_stress_7d else np.nan
            stress_trend = (stress_last - stress_mean7) if (pd.notna(stress_last) and pd.notna(stress_mean7)) else np.nan
            aff_conf = 0.85 if pd.notna(stress_last) else np.nan

            # Effective baseline values for targets
            eff_sleep = sleep_h if pd.notna(sleep_h) else 7.0
            eff_wake = wake_h if pd.notna(wake_h) else DEFAULT_WAKE_H
            eff_stress_7d = stress_mean7 if pd.notna(stress_mean7) else pss
            eff_stress_last = stress_last if pd.notna(stress_last) else eff_stress_7d
            sleep_debt = max(0.0, 7.5 - eff_sleep)

            # PAM arousal around this day
            pam_today = [
                p["arousal"] for p in u_pam_list
                if abs((p["dt"].date() - curr_date).days) <= 1
            ]
            self_arousal = float(np.mean(pam_today)) if pam_today else 0.5

            # Productivity self-report
            self_prod = ema_prod.get((uid, d_str), 0.6)

            # Build request features dictionary
            req_features = {
                "sleep_h": sleep_h,
                "wake_hour": wake_h,
                "stress_last": stress_last,
                "stress_mean7": stress_mean7,
                "stress_trend": stress_trend,
                "aff_conf": aff_conf,
                "steps_24h": steps_24h,
                "tasks_due_24h": tasks_due_24h,
                "minutes_due_24h": minutes_due_24h,
                "tasks_overdue": tasks_overdue,
                "tasks_pending": tasks_pending,
                "asof_hour": 9.0,
            }

            # Generate 24 hourly forecast slots
            for k in range(24):
                slot_clock = (9.0 + k) % 24
                sf = slot_features(slot_clock, eff_wake, float(k))

                # Circadian alertness and homeostatic fatigue
                c = circadian(slot_clock)
                s = homeostatic_fatigue(sf["hours_awake"])
                debt = clip01(sleep_debt / 7.5)
                wl_minutes = clip01(minutes_due_24h / 480.0)
                wl_overdue = clip01(tasks_overdue / 5.0)
                stress_val = eff_stress_last
                stress_7d_val = eff_stress_7d

                # Energy: circadian alertness, homeostatic sleep pressure, debt, stress, steps, PAM arousal
                base_energy = clip01(
                    0.20 + 0.80 * c - 0.35 * s - 0.25 * debt - 0.10 * stress_val
                    + min(0.05, steps_24h / 150000.0)
                )
                energy = clip01(0.70 * base_energy + 0.30 * self_arousal + rng.normal(0, 0.015))

                # Mental fatigue: rises with wakefulness, sleep debt, workload, stress, trouble staying awake
                fatigue = clip01(
                    0.15 + 0.55 * s + 0.25 * debt + 0.15 * wl_minutes + 0.15 * stress_val
                    + 0.05 * (social_trouble - 1) - 0.10 * (c - 0.5)
                    + rng.normal(0, 0.015)
                )

                # Focus: cognitive alertness, self-reported productivity, penalized by fatigue and stress
                base_focus = clip01(0.60 * energy + 0.40 * (1 - fatigue) - 0.15 * stress_val - 0.05 * wl_minutes)
                focus = clip01(0.75 * base_focus + 0.25 * self_prod + rng.normal(0, 0.015))

                # Burnout risk: chronic stress, survey baseline, overdue tasks, debt, fatigue
                burnout = clip01(
                    0.08 + 0.35 * stress_7d_val + 0.20 * burnout_base + 0.20 * wl_overdue
                    + 0.15 * wl_minutes + 0.15 * debt + 0.10 * fatigue
                    + rng.normal(0, 0.01)
                )

                rows.append({
                    "user_id": uid,
                    "date": d_str,
                    **req_features,
                    **sf,
                    "mental_fatigue": round(fatigue, 4),
                    "focus": round(focus, 4),
                    "energy": round(energy, 4),
                    "burnout_risk": round(burnout, 4),
                })

    df = pd.DataFrame(rows)
    print(f"Generated dataset with shape: {df.shape}")

    if output_csv:
        out_path = pathlib.Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Saved dataset to {out_path.resolve()}")

    return df


def get_data_dir() -> pathlib.Path:
    if os.getenv("DATA_DIR"):
        return pathlib.Path(os.getenv("DATA_DIR")).resolve()
    candidates = [
        pathlib.Path(__file__).resolve().parents[4] / "07-datasets",
        pathlib.Path("c:/UNI/4Y 1S/RP/Project/07-datasets"),
        pathlib.Path("../07-datasets").resolve(),
        pathlib.Path("../../07-datasets").resolve(),
        pathlib.Path("../../../07-datasets").resolve(),
    ]
    for c in candidates:
        if c.exists():
            return c
    return pathlib.Path("c:/UNI/4Y 1S/RP/Project/07-datasets")


def load_or_build_dataset(data_dir: str | pathlib.Path | None = None) -> pd.DataFrame:
    """Load cached StudentLife training dataset or build it if not found."""
    if data_dir is None:
        base = get_data_dir()
    else:
        base = pathlib.Path(data_dir)

    cache_file = base / "m2_studentlife.csv"
    if cache_file.exists():
        print(f"Loading cached dataset from {cache_file}...")
        return pd.read_csv(cache_file)

    ds_dir = base / "dataset" if (base / "dataset").exists() else base
    return build_studentlife_dataset(ds_dir, output_csv=cache_file)


if __name__ == "__main__":
    df = load_or_build_dataset()
    print("Dataset summary:")
    print(df[TARGETS].describe().T[["mean", "std", "min", "max"]])
