"""The three supporting datasets of Module 3, turned into small files the scheduler and demo use.

  PMData (Thambawita et al., 2020)  -> models/pmdata_wellness_days.json
      Real daily self-reports from 16 adults (readiness 0-10; fatigue, mood, stress, sleep quality 1-5, higher =
      better). Used as a stand-in for Module 2's wellness forecast until Module 2 is connected: each day becomes
      hourly energy / focus / fatigue / burnout values (daily level x a typical time-of-day alertness curve).
  ATUS 2025 (US Bureau of Labor Statistics) -> models/atus_day_profiles.json
      Real 24-hour diaries. For weekday diaries of working adults and of university students, the weighted share
      of people who are asleep, at work/class or travelling in each 15 minutes of the day. Quarter-hours when at
      least 40% are at work, in class or travelling become the default busy blocks of a new user, until they add
      their own calendar.
  StudentLife (Wang et al., 2014)   -> reports/studentlife_deadlines_stress.json
      48 students over a 10-week term: number of deadlines per day and self-reported stress and sleep. Measures how
      much stress rises and sleep falls when deadlines pile up: the evidence for Module 3's "wellness affects task
      completion" link and the effect sizes for the simulator.

Usage (from services/m3-schedule-optimizer):
    PYTHONPATH=src python -m m3.context_datasets            # build all three, save figures, log to MLflow
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .config import DATA_DIR, MODELS_DIR, MODULE_ROOT, REPORTS_DIR

# Typical alertness across the day (0-1): low at night, rising in the morning, peak late morning, post-lunch dip,
# second peak late afternoon, decline in the evening. A standard circadian shape, used only to spread a day's
# wellness level over its hours until Module 2 supplies real intraday forecasts.
CIRCADIAN = [0.15, 0.1, 0.08, 0.08, 0.1, 0.2, 0.4, 0.6, 0.75, 0.88, 0.95, 0.97,
             0.9, 0.78, 0.74, 0.8, 0.88, 0.9, 0.85, 0.75, 0.65, 0.5, 0.35, 0.22]  # fmt: skip


# ----------------------------------------------------------------------------- PMData
def build_pmdata(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    base = Path(data_dir) / "pmdata"
    files = sorted(base.glob("raw/**/p*/pmsys/wellness.csv")) or sorted(base.glob("**/pmsys/wellness.csv"))
    if not files:
        raise FileNotFoundError(f"no pmsys/wellness.csv under {base}")
    frames = []
    for f in files:
        d = pd.read_csv(f)
        d["participant"] = f.parent.parent.name
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    for c in ["fatigue", "mood", "readiness", "sleep_quality", "stress", "sleep_duration_h", "soreness"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").replace(0, np.nan)  # 0 = not answered
    d["date"] = pd.to_datetime(d.effective_time_frame, utc=True).dt.tz_convert("Europe/Oslo").dt.date.astype(str)
    d = d.groupby(["participant", "date"], as_index=False).mean(numeric_only=True)
    s15 = lambda c: (d[c] - 1) / 4  # noqa: E731  -- 1-5 scale to 0-1, higher = better
    d["energy"] = pd.concat([d.readiness / 10, s15("fatigue"), s15("sleep_quality")], axis=1).mean(axis=1)
    d["focus"] = pd.concat([d.readiness / 10, s15("mood"), s15("stress")], axis=1).mean(axis=1)
    d["calm"] = s15("stress")
    d = d.dropna(subset=["energy", "focus"])
    return d[["participant", "date", "readiness", "fatigue", "mood", "stress", "sleep_quality", "sleep_duration_h",
              "energy", "focus", "calm"]].round(3)  # fmt: skip


def hourly_wellness(day: dict, hour: int) -> dict:
    """Spread one day's level over its hours -> the four Module 2 wellness scores (0-1)."""
    c = CIRCADIAN[hour % 24]
    energy = float(np.clip(0.15 + 0.85 * day["energy"] * c / 0.97, 0, 1))
    focus = float(np.clip(0.15 + 0.85 * day["focus"] * c / 0.97, 0, 1))
    calm = day.get("calm")
    calm = 0.5 if calm is None or (isinstance(calm, float) and np.isnan(calm)) else calm
    return {
        "energy": round(energy, 3),
        "focus": round(focus, 3),
        "mental_fatigue": round(1 - energy, 3),
        "burnout_risk": round(float(np.clip(0.6 * (1 - calm) + 0.4 * (1 - day["energy"]), 0, 1)), 3),
    }


def forecast_from_days(days: list[dict], user_id: str, start: datetime, hours: int = 72) -> dict:
    """A Module 2-format WellnessForecast built from consecutive PMData days (one per calendar day)."""
    slots = []
    start = start.replace(minute=0, second=0, microsecond=0)
    for h in range(hours):
        t = start + timedelta(hours=h)
        day = days[min(len(days) - 1, (t.date() - start.date()).days)]
        slots.append(
            {"start": t.isoformat(), "end": (t + timedelta(hours=1)).isoformat(), **hourly_wellness(day, t.hour)}
        )
    first = days[0]
    sleep = first.get("sleep_duration_h")
    return {
        "user_id": user_id,
        "generated_at": start.astimezone(ZoneInfo("UTC")).isoformat(),
        "slot_minutes": 60,
        "slots": slots,
        "sleep_hours_last_night": None if sleep is None or (isinstance(sleep, float) and np.isnan(sleep)) else sleep,
        "model": {"name": "m2-standin-pmdata", "version": "0.1.0"},
    }


# ----------------------------------------------------------------------------- ATUS
BUSY_SHARE = 0.4  # a quarter-hour is a default busy block when at least 40% of people are at work, class or travelling


def build_atus(data_dir: Path = DATA_DIR) -> dict:
    ext = Path(data_dir) / "atus" / "raw" / "extracted"
    act = pd.read_csv(next(ext.glob("atusact_*.dat")), usecols=["TUCASEID", "TUSTARTTIM", "TUSTOPTIME", "TRCODE"])
    resp = pd.read_csv(
        next(ext.glob("atusresp_*.dat")),
        usecols=["TUCASEID", "TELFS", "TESCHENR", "TESCHLVL", "TUDIARYDAY", "TRHOLIDAY", "TUFINLWGT"],
    )
    resp = resp[resp.TUDIARYDAY.between(2, 6) & (resp.TRHOLIDAY != 1)]  # Monday-Friday, not a holiday
    personas = {
        "working_adult": resp[(resp.TELFS == 1) & (resp.TESCHENR != 1)],
        "university_student": resp[(resp.TESCHENR == 1) & (resp.TESCHLVL == 2)],
    }
    code = act.TRCODE.astype(int)
    act["kind"] = np.select(
        [code // 100 == 101, code // 10000 == 5, code // 10000 == 6, code // 10000 == 18],
        ["sleep", "work", "class", "travel"],
        "free",
    )

    def minutes(s):
        h, m, _ = (int(x) for x in s.split(":"))
        return h * 60 + m

    act["s"] = act.TUSTARTTIM.map(minutes)
    act["e"] = act.TUSTOPTIME.map(minutes)
    out = {
        "source": "ATUS 2025 activity + respondent files, weekday non-holiday diaries, survey-weighted",
        "personas": {},
    }
    for name, r in personas.items():
        a = act[act.TUCASEID.isin(r.TUCASEID)]
        w = r.set_index("TUCASEID").TUFINLWGT
        grid = {k: np.zeros(96) for k in ("sleep", "work", "class", "travel", "free")}
        for row in a.itertuples(index=False):
            s, e = row.s, row.e if row.e > row.s else row.e + 1440  # diary runs 4am -> 4am
            for q in range(s // 15, min(e // 15, s // 15 + 96)):
                grid[row.kind][q % 96] += w[row.TUCASEID]
        total = sum(grid.values())
        shares = {k: (v / np.where(total > 0, total, 1)).round(3).tolist() for k, v in grid.items()}
        # daytime commitments (sleep is handled by the scheduler's day window): work, class and travel
        committed = np.array(shares["work"]) + np.array(shares["class"]) + np.array(shares["travel"])
        busy = [q for q in range(96) if committed[q] >= BUSY_SHARE]
        blocks, start = [], None
        for q in range(97):
            inside = q in busy
            if inside and start is None:
                start = q
            if not inside and start is not None:
                blocks.append([f"{start // 4:02d}:{start % 4 * 15:02d}", f"{q // 4 % 24:02d}:{q % 4 * 15:02d}"])
                start = None
        out["personas"][name] = {"n_diaries": len(r), "shares_15min": shares, "default_busy_blocks": blocks}
    return out


# ----------------------------------------------------------------------------- StudentLife
STRESS_LEVEL = {1: 3, 2: 4, 3: 5, 4: 2, 5: 1}  # EMA answer -> stress 1 (feeling great) .. 5 (stressed out)


def build_studentlife(data_dir: Path = DATA_DIR) -> dict:
    base = Path(data_dir) / "studentlife" / "raw"
    base = base / "dataset" if (base / "dataset").exists() else base
    tz = ZoneInfo("America/New_York")
    dl = pd.read_csv(base / "education" / "deadlines.csv").melt(id_vars="uid", var_name="date", value_name="deadlines")
    rows = []
    for kind, key in (("stress", "level"), ("sleep", "hour")):
        for f in sorted((base / "EMA" / "response" / kind.capitalize()).glob("*.json")):
            uid = f.stem.split("_")[-1]
            for r in json.loads(f.read_text()):
                if key in r and str(r[key]).isdigit():
                    v = int(r[key])
                    val = STRESS_LEVEL.get(v) if kind == "stress" else 3 + (v - 1) * 0.5
                    if val is None:
                        continue
                    d = datetime.fromtimestamp(int(r["resp_time"]), tz).date().isoformat()
                    rows.append({"uid": uid, "date": d, "kind": kind, "value": val})
    ema = (
        pd.DataFrame(rows)
        .pivot_table(index=["uid", "date"], columns="kind", values="value", aggfunc="mean")
        .reset_index()
    )
    dl = dl.sort_values(["uid", "date"])
    # deadline load = deadlines today and in the next 2 days
    dl["load3"] = dl.groupby("uid").deadlines.transform(lambda s: s[::-1].rolling(3, min_periods=1).sum()[::-1])
    m = ema.merge(dl, on=["uid", "date"], how="inner")
    res = {
        "n_students": int(m.uid.nunique()),
        "n_student_days": len(m),
        "source": "StudentLife 2013 (Wang et al., UbiComp 2014)",
    }
    for kind in ("stress", "sleep"):
        d = m.dropna(subset=[kind, "load3"]).copy()
        d["load_bucket"] = pd.cut(d.load3, [-1, 0, 1, 2, 100], labels=["0", "1", "2", "3+"])
        by = d.groupby("load_bucket", observed=True)[kind].agg(["mean", "count"]).round(3)
        # within-student effect: centre each student's values on their own mean, then fit a line
        dx = d.load3 - d.groupby("uid").load3.transform("mean")
        dy = d[kind] - d.groupby("uid")[kind].transform("mean")
        slope = float(np.polyfit(dx, dy, 1)[0]) if dx.std() > 0 else float("nan")
        r = float(np.corrcoef(dx, dy)[0, 1]) if dx.std() > 0 else float("nan")
        res[kind] = {
            "by_deadline_load": {str(k): {"mean": float(v["mean"]), "n": int(v["count"])} for k, v in by.iterrows()},
            "within_student_slope_per_deadline": round(slope, 4),
            "within_student_r": round(r, 4),
            "n": len(d),
        }
    return res


# ----------------------------------------------------------------------------- figures + main
def _figures(pm: pd.DataFrame, atus: dict, sl: dict) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    # PMData: one good and one hard day spread over the hours
    good = pm.loc[pm.energy.idxmax()].to_dict()
    bad = pm.loc[pm.energy.idxmin()].to_dict()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    hrs = range(24)
    ax.plot(
        hrs,
        [hourly_wellness(good, h)["energy"] for h in hrs],
        lw=2,
        label=f"good day ({good['participant']} {good['date']})",
    )
    ax.plot(
        hrs,
        [hourly_wellness(bad, h)["energy"] for h in hrs],
        lw=2,
        label=f"hard day ({bad['participant']} {bad['date']})",
    )
    ax.set_xlabel("hour of day")
    ax.set_ylabel("energy (0-1)")
    ax.set_title("PMData daily reports -> hourly energy (stand-in for Module 2)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out["pmdata"] = REPORTS_DIR / "fig5_pmdata_wellness_standin.png"
    fig.savefig(out["pmdata"], dpi=150)
    # ATUS: committed share by time of day
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    x = np.arange(96) / 4
    for ax, (name, p) in zip(axs, atus["personas"].items(), strict=True):
        sh = p["shares_15min"]
        ax.stackplot(x, sh["sleep"], sh["work"], sh["class"], sh["travel"], sh["free"],
                     labels=["sleep", "work", "class", "travel", "free"], alpha=0.85)  # fmt: skip
        ax.set_title(f"{name.replace('_', ' ')} (n={p['n_diaries']:,} weekday diaries)", fontsize=10)
        ax.set_xlabel("hour of day")
        ax.set_xlim(0, 24)
    axs[0].set_ylabel("share of people")
    axs[1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    fig.suptitle("ATUS 2025: what people are doing across a weekday")
    fig.tight_layout()
    out["atus"] = REPORTS_DIR / "fig6_atus_day_profiles.png"
    fig.savefig(out["atus"], dpi=150)
    # StudentLife: stress and sleep by deadline load
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, kind, lab in ((axs[0], "stress", "stress (1 great .. 5 stressed out)"), (axs[1], "sleep", "sleep hours")):
        b = sl[kind]["by_deadline_load"]
        bars = ax.bar(list(b), [v["mean"] for v in b.values()], color="#4C78A8")
        ax.bar_label(bars, [f"{v['mean']:.2f}\n(n={v['n']})" for v in b.values()], fontsize=8)
        ax.set_xlabel("deadlines today + next 2 days")
        ax.set_ylabel(lab)
        lo = min(v["mean"] for v in b.values())
        ax.set_ylim(lo * 0.85, max(v["mean"] for v in b.values()) * 1.12)
    fig.suptitle("StudentLife: more deadlines -> more stress, less sleep")
    fig.tight_layout()
    out["studentlife"] = REPORTS_DIR / "fig7_studentlife_deadlines.png"
    fig.savefig(out["studentlife"], dpi=150)
    plt.close("all")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracking-uri", default=f"sqlite:///{MODULE_ROOT / 'mlflow.db'}")
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    pm = build_pmdata()
    (MODELS_DIR / "pmdata_wellness_days.json").write_text(pm.to_json(orient="records"))
    print(f"PMData: {len(pm):,} participant-days from {pm.participant.nunique()} people")
    atus = build_atus()
    (MODELS_DIR / "atus_day_profiles.json").write_text(json.dumps(atus))
    for k, v in atus["personas"].items():
        print(f"ATUS {k}: {v['n_diaries']:,} diaries; default busy blocks {v['default_busy_blocks']}")
    sl = build_studentlife()
    (REPORTS_DIR / "studentlife_deadlines_stress.json").write_text(json.dumps(sl, indent=2))
    print(f"StudentLife: {sl['n_students']} students, {sl['n_student_days']} student-days; "
          f"stress +{sl['stress']['within_student_slope_per_deadline']} per deadline (r={sl['stress']['within_student_r']}), "
          f"sleep {sl['sleep']['within_student_slope_per_deadline']} h per deadline")  # fmt: skip
    figs = _figures(pm, atus, sl)
    if a.no_mlflow:
        return
    import mlflow

    mlflow.set_tracking_uri(a.tracking_uri)
    mlflow.set_experiment("M3 - Supporting datasets (PMData, ATUS, StudentLife)")
    with mlflow.start_run(run_name="pmdata-wellness-standin"):
        mlflow.set_tag(
            "mlflow.note.content",
            "PMData daily wellness -> hourly Module 2-format forecast (stand-in until Module 2 is connected).",
        )
        mlflow.log_params({"dataset": "PMData", "participants": int(pm.participant.nunique()), "days": len(pm)})
        mlflow.log_metrics({"mean_energy": float(pm.energy.mean()), "mean_focus": float(pm.focus.mean())})
        mlflow.log_artifact(str(figs["pmdata"]), "plots")
        mlflow.log_artifact(str(MODELS_DIR / "pmdata_wellness_days.json"), "data")
    with mlflow.start_run(run_name="atus-day-profiles"):
        mlflow.set_tag(
            "mlflow.note.content",
            "ATUS 2025 weekday diaries -> when working adults and students are busy (default busy blocks for new users and simulated users).",
        )
        mlflow.log_params(
            {"dataset": "ATUS 2025", **{f"diaries_{k}": v["n_diaries"] for k, v in atus["personas"].items()}}
        )
        mlflow.log_artifact(str(figs["atus"]), "plots")
        mlflow.log_artifact(str(MODELS_DIR / "atus_day_profiles.json"), "data")
    with mlflow.start_run(run_name="studentlife-deadlines-vs-stress"):
        mlflow.set_tag(
            "mlflow.note.content",
            "StudentLife: how stress and sleep change when deadlines pile up (evidence for the wellness link and effect sizes for the simulator).",
        )
        mlflow.log_params(
            {"dataset": "StudentLife 2013", "students": sl["n_students"], "student_days": sl["n_student_days"]}
        )
        mlflow.log_metrics({
            "stress_slope_per_deadline": sl["stress"]["within_student_slope_per_deadline"],
            "stress_within_r": sl["stress"]["within_student_r"],
            "sleep_slope_per_deadline": sl["sleep"]["within_student_slope_per_deadline"],
            "sleep_within_r": sl["sleep"]["within_student_r"],
        })  # fmt: skip
        mlflow.log_artifact(str(figs["studentlife"]), "plots")
        mlflow.log_artifact(str(REPORTS_DIR / "studentlife_deadlines_stress.json"), "data")


if __name__ == "__main__":
    main()
