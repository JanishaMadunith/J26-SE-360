"""Small preview of the prepared (model-ready) data built from each dataset, for the demo's "Prepared data" tab.

Writes models/prepared_preview.json (~100 KB): for OULAD, MS-LaTTE, PMData, ATUS and StudentLife: where the raw data
comes from, the preparation steps, a few numbers, and real sample rows of the prepared table.

Usage (from services/m3-schedule-optimizer):
    PYTHONPATH=src python -m m3.build_prepared_preview [--samples-cache path/to/oulad_samples.pkl]
"""

import argparse
import json

import numpy as np
import pandas as pd

from .config import MODELS_DIR, REPORTS_DIR, SEED, TEST_PRESENTATION, TRAIN_PRESENTATIONS
from .context_datasets import hourly_wellness
from .tasktime import PARTS, dataset, load_mslatte, votes

VALIDATION = json.loads((MODELS_DIR / "model_card.json").read_text()).get("validated_on")


def clean(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return round(float(v), 3)
    return v if isinstance(v, (int, bool)) else str(v)


def table(df: pd.DataFrame) -> dict:
    return {"columns": list(df.columns), "rows": [[clean(v) for v in r] for r in df.itertuples(index=False)]}


def oulad(df: pd.DataFrame, features: list[str]) -> dict:
    show = [
        "code_module",
        "code_presentation",
        "id_student",
        "id_assessment",
        "assessment_type",
        "H",
        "weight",
        "prior_on_time_rate",
        "n_prior",
        "days_since_last",
        "active_7",
        "clicks_7",
        "n_prior_outstanding",
        "n_overdue_all",
        "n_upcoming_7_all",
        "y",
    ]
    rng = np.random.default_rng(SEED)
    picks = []
    for h in sorted(df.H.unique(), reverse=True):  # 3 rows per horizon, both outcomes
        g = df[h == df.H]
        for y in (1, 0):
            gg = g[g.y == y]
            picks += list(gg.index[rng.choice(len(gg), size=2 if y == 1 else 1, replace=False)])
    sample = df.loc[picks, show].copy()
    sample["y"] = sample.y.map({1: "1 on time", 0: "0 late / missing"})
    split = {
        p: (
            "validation, then final training"
            if p == VALIDATION
            else "train"
            if p in TRAIN_PRESENTATIONS
            else "test"
            if p == TEST_PRESENTATION
            else "other"
        )
        for p in sorted(df.code_presentation.unique())
    }
    one = df.loc[picks[0], features]
    return {
        "name": "OULAD → task-completion cases",
        "used_for": "Completion model (will this task be done on time?)",
        "raw": "Open University Learning Analytics Dataset: 7 tables (students, courses, assessments, submissions, "
        "10.6 M daily click records) for 32,593 students in 22 course runs",
        "steps": [
            "Each assessment with a deadline becomes a task",
            "Asked at 14, 7, 3 and 1 day(s) before the deadline (t = deadline − H)",
            "Only data up to day t is used (clicks, earlier submissions) — no peeking into the future",
            "Tasks already handed in by day t are dropped (nothing left to predict)",
            "65 features: past on-time rate, recent activity, backlog, overdue work, other deadlines, task size",
            "Label y = 1 if submitted on or before the deadline, else 0 (late, never, or withdrew)",
        ],
        "stats": {
            "prepared rows (cases)": len(df),
            "features per case": len(features),
            "students": int(df.id_student.nunique()),
            "on time (y = 1)": f"{df.y.mean():.1%}",
            **{f"rows {p} ({s})": int((df.code_presentation == p).sum()) for p, s in split.items()},
        },
        "per_horizon": {
            f"{int(h)} day(s) before": {"rows": len(g), "on time": round(float(g.y.mean()), 3)}
            for h, g in sorted(df.groupby("H"), key=lambda x: -x[0])
        },
        "sample": table(sample),
        "one_row_all_features": {c: clean(v) for c, v in one.items()},
    }


def mslatte() -> dict:
    raw = load_mslatte()
    texts, y, _ = dataset()
    rows = []
    for t in raw:
        v = votes(t)
        title = (t.get("TaskTitle") or "").strip()
        if v.sum() == 0 or not title:
            continue
        rows.append(
            {
                "task title": title,
                "list": t.get("ListTitle"),
                **{f"{p} votes": round(float(x), 2) for p, x in zip(PARTS, v, strict=True)},
                "label (most votes)": PARTS[int(np.argmax(v))] + (" (tie)" if np.sort(v)[-1] == np.sort(v)[-2] else ""),
                "model input": None,
            }
        )
    for r, tx in zip(rows, texts, strict=True):
        r["model input"] = tx
    rng = np.random.default_rng(SEED)
    sample = pd.DataFrame([rows[i] for i in rng.choice(len(rows), 14, replace=False)])
    counts = pd.Series([PARTS[i] for i in y]).value_counts()
    return {
        "name": "MS-LaTTE → labelled to-do titles",
        "used_for": "Time-of-day model (best part of the day for a task)",
        "raw": "Microsoft to-do titles (10,101), each judged by 5 annotators: when would people do this "
        "(weekday/weekend × morning/afternoon/evening/night/anytime)",
        "steps": [
            "Votes pooled per part of day (weekday + weekend); 'anytime' spread evenly over the four parts",
            "Label = the part of day with most votes; vote shares kept for top-2 and agreement scores",
            "Model input = lower-case title + the list it is in (in the app: the task category)",
            "Same 80/20 test split for every model version; 5-fold cross-validation",
        ],
        "stats": {
            "prepared rows (tasks)": len(texts),
            **{f"label {p}": int(counts.get(p, 0)) for p in PARTS},
        },
        "sample": table(sample),
    }


def pmdata() -> dict:
    days = json.loads((MODELS_DIR / "pmdata_wellness_days.json").read_text())
    df = pd.DataFrame(days)
    rng = np.random.default_rng(SEED)
    sample = df.iloc[sorted(rng.choice(len(df), 12, replace=False))]
    ex = df.iloc[len(df) // 2].to_dict()
    hourly = pd.DataFrame([{"hour": f"{h:02d}:00", **hourly_wellness(ex, h)} for h in range(6, 24)])
    return {
        "name": "PMData → daily wellness → hourly forecast",
        "used_for": "Stand-in for Module 2's wellness forecast (energy, focus, fatigue, burnout per hour)",
        "raw": "PMData (Thambawita et al., 2020): 16 people, 5 months of Fitbit + daily self-reports "
        "(readiness, fatigue, mood, stress, sleep)",
        "steps": [
            "One row per person-day with the self-report scores and Fitbit sleep",
            "energy = mean of readiness, fatigue and sleep quality; focus = mean of readiness, mood and stress; "
            "calm = stress (pmsys 1–5 scales, where higher = better, mapped to 0–1)",
            "Each day is spread over the hours with a standard alertness curve (high mid-morning, dip after lunch)",
            "Output has exactly Module 2's WellnessForecast format, so the real Module 2 can replace it",
        ],
        "stats": {
            "prepared rows (person-days)": len(df),
            "people": int(df.participant.nunique()),
            "mean energy": round(float(df.energy.mean()), 3),
            "mean focus": round(float(df.focus.mean()), 3),
        },
        "sample": table(sample),
        "example_day": {"participant": ex["participant"], "date": ex["date"], "hourly": table(hourly)},
    }


def atus() -> dict:
    a = json.loads((MODELS_DIR / "atus_day_profiles.json").read_text())
    rows = []
    for name, p in a["personas"].items():
        sh = p["shares_15min"]
        for h in range(6, 24):
            q = slice(h * 4, h * 4 + 4)
            r = {"persona": name.replace("_", " "), "hour": f"{h:02d}:00"}
            for k in ("work", "class", "travel", "sleep"):
                if k in sh:
                    r[k] = round(float(np.mean(sh[k][q])), 2)
            busy = sum(r.get(k, 0) for k in ("work", "class", "travel"))
            r["busy share"] = round(busy, 2)
            r["blocked?"] = "yes" if busy >= 0.4 else ""
            rows.append(r)
    return {
        "name": "ATUS 2025 → typical busy hours",
        "used_for": "Default work / class hours per persona (until the user's own calendar is connected)",
        "raw": "American Time Use Survey 2025: what each respondent did, minute by minute, for one day",
        "steps": [
            "Weekday, non-holiday diaries; working adults and university students",
            "Survey-weighted share of people at work, in class or travelling, per 15 minutes",
            "A quarter-hour is a default busy block when at least 40% of that group is at work / class / travel",
        ],
        "stats": {
            **{f"diaries {k.replace('_', ' ')}": int(v["n_diaries"]) for k, v in a["personas"].items()},
            **{
                f"busy blocks {k.replace('_', ' ')}": ", ".join("–".join(b) for b in v["default_busy_blocks"])
                for k, v in a["personas"].items()
            },
        },
        "sample": table(pd.DataFrame(rows)),
    }


def studentlife() -> dict:
    s = json.loads((REPORTS_DIR / "studentlife_deadlines_stress.json").read_text())
    rows = [
        {
            "deadlines in next 3 days": k,
            "mean stress (1–5)": s["stress"]["by_deadline_load"][k]["mean"],
            "student-days (stress)": s["stress"]["by_deadline_load"][k]["n"],
            "mean sleep (h)": s["sleep"]["by_deadline_load"].get(k, {}).get("mean"),
            "student-days (sleep)": s["sleep"]["by_deadline_load"].get(k, {}).get("n"),
        }
        for k in s["stress"]["by_deadline_load"]
    ]
    return {
        "name": "StudentLife → deadline effect on stress and sleep",
        "used_for": "Lowers the wellness forecast when deadlines pile up",
        "raw": "StudentLife (Wang et al., UbiComp 2014): 48 Dartmouth students, 10 weeks of phone sensing, "
        "stress and sleep surveys, and their real deadlines",
        "steps": [
            "One row per student-day: stress report, sleep, number of deadlines in the next 3 days",
            "Within-student comparison (each student against their own average), so personalities cancel out",
            "Slope per extra deadline used by the scheduler",
        ],
        "stats": {
            "students": s["n_students"],
            "student-days": s["n_student_days"],
            "stress per extra deadline": f"+{s['stress']['within_student_slope_per_deadline']:.2f}",
            "sleep per extra deadline": f"{s['sleep']['within_student_slope_per_deadline']:.2f} h",
        },
        "sample": table(pd.DataFrame(rows)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-cache", default=None)
    a = ap.parse_args()
    if a.samples_cache:
        df = pd.read_pickle(a.samples_cache)
    else:
        from .data_oulad import load_oulad
        from .features_oulad import build_dataset

        print("Building the OULAD samples from 07-datasets (a few minutes)...")
        df = build_dataset(load_oulad())
    features = json.loads((MODELS_DIR / "model_card.json").read_text())["features"]
    out = {"datasets": [oulad(df, features), mslatte(), pmdata(), atus(), studentlife()]}
    path = MODELS_DIR / "prepared_preview.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
