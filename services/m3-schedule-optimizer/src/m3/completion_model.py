"""Serving wrapper for the trained task-completion model (XGBoost, trained on OULAD by m3.train_oulad).

* `CompletionModel.predict(features)` returns P(task finished on time) for one or more feature rows. Any of the
  65 model features may be left out; left-out ones take the training-set median (or mode for categories), stored in
  models/completion_defaults.json by m3.build_demo_assets. A feature given as None means "unknown" (for example
  no task history yet); XGBoost handles that natively, as it did in training.
* `CompletionModel.explain(features)` returns the top reasons (exact per-prediction SHAP contributions from
  XGBoost) in plain English.
* `BehaviourProfile` + `features_from_app()` map what the app knows about a user and a task onto the model's
  inputs. The app's task log provides the same kinds of signal as OULAD: past on-time rate, recent activity,
  unfinished and overdue work, time left before the deadline, task size and importance.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from .config import MODELS_DIR

# Plain-English names for the clues people ask about. Anything not listed falls back to its feature name.
LABELS = {
    "H": "Days left before the deadline",
    "is_tma": "Task size (big assignment vs. short quiz)",
    "weight": "How much the task counts",
    "prior_same_type_on_time_rate": "Past on-time rate for this kind of task",
    "prior_on_time_rate": "Past on-time rate (all tasks)",
    "days_since_last": "Days since last active",
    "n_prior_outstanding": "Earlier tasks still unfinished",
    "n_overdue_all": "Overdue tasks across all courses",
    "last_prior_days_early": "How early/late the last task was handed in",
    "prior_mean_days_early": "Usually hands in early or late",
    "last_prior_on_time": "Was the last task on time",
    "clicks_30": "Activity in the last 30 days",
    "clicks_14": "Activity in the last 14 days",
    "clicks_7": "Activity in the last 7 days",
    "clicks_3": "Activity in the last 3 days",
    "active_7": "Active days in the last 7 days",
    "active_14": "Active days in the last 14 days",
    "rel_clicks_7": "Recent activity vs. own average",
    "self_rel_clicks_7": "Recent activity vs. own usual level",
    "pct_clicks_7": "Recent activity vs. classmates",
    "pct_clicks_14": "2-week activity vs. classmates",
    "pct_clicks_total": "Overall activity vs. classmates",
    "clicks_since_prev_deadline": "Work done since the previous deadline",
    "active_since_prev_deadline": "Active days since the previous deadline",
    "n_upcoming_7_all": "Other deadlines in the next 7 days",
    "n_upcoming_14_all": "Other deadlines in the next 14 days",
    "studied_credits": "Total study load",
    "n_prior_same_type": "Number of similar tasks done before",
    "n_prior": "Number of tasks done before",
    "n_prior_on_time": "Number of earlier tasks done on time",
    "n_prior_late_known": "Number of earlier tasks done late",
    "n_future_done": "Later tasks already done",
    "a_order": "Position of the task in the course",
    "gap_prev": "Days since the previous deadline",
    "gap_next": "Days until the next deadline",
    "date_registration": "How early the student registered",
    "days_since_registration": "Days since starting",
    "site_visits_7": "Visits in the last 7 days",
    "site_visits_14": "Visits in the last 14 days",
    "trend_7": "Activity trend this week",
    "code_module": "Course",
    "deadline_frac": "How far into the course the task is",
}
# Map app priority (1-5) to an OULAD-style weight in percent (median OULAD weight is 10).
PRIORITY_WEIGHT = {1: 0.0, 2: 5.0, 3: 10.0, 4: 18.0, 5: 25.0}


@dataclass
class BehaviourProfile:
    """What the app knows about the user's recent behaviour (from the task-event log)."""

    on_time_rate: float | None = None  # all tasks, 0-1
    same_type_on_time_rate: float | None = None  # this task category, 0-1
    tasks_done_before: int = 4
    days_since_active: float = 1.0
    active_days_7: int = 4  # days with any app activity in the last 7 days
    unfinished_earlier_tasks: int = 0
    overdue_tasks: int = 0
    upcoming_deadlines_7d: int = 0
    last_task_on_time: bool | None = None


def features_from_app(
    profile: BehaviourProfile, days_to_deadline: float, estimated_minutes: int, priority: int
) -> dict:
    """Translate an app task + behaviour profile into the model's inputs (others use training medians)."""
    rate = profile.on_time_rate
    same = profile.same_type_on_time_rate if profile.same_type_on_time_rate is not None else rate
    n = max(0, profile.tasks_done_before) if rate is not None else 0  # no history yet
    active7 = max(0, min(7, profile.active_days_7))
    f = {
        "H": float(min(max(days_to_deadline, 0.0), 14.0)),
        # App tasks are things the user committed to, like OULAD's tutor-marked assignments. OULAD's computer-marked
        # quizzes follow a course-specific bulk-submission pattern that does not carry over, so it is not used.
        "is_tma": 1,
        "weight": PRIORITY_WEIGHT.get(int(priority), 10.0),
        "n_prior": float(n),
        "n_prior_same_type": float(max(0, n // 2)),
        "prior_on_time_rate": None if rate is None else float(rate),
        "prior_same_type_on_time_rate": None if same is None else float(same),
        "n_prior_on_time": None if rate is None else float(round(rate * n)),
        "days_since_last": float(max(0.0, profile.days_since_active)),
        "active_7": float(active7),
        "active_14": float(min(14, active7 * 2)),
        "n_prior_outstanding": float(max(0, profile.unfinished_earlier_tasks)),
        "n_overdue_all": float(max(0, profile.overdue_tasks)),
        "n_upcoming_7_all": float(max(0, profile.upcoming_deadlines_7d)),
        "last_prior_on_time": (
            float(profile.last_task_on_time)
            if profile.last_task_on_time is not None
            else (None if rate is None else float(rate >= 0.5))
        ),
    }
    # activity volume scaled from active days (OULAD median: ~8 clicks per active day)
    f["clicks_7"] = active7 * 8.0
    f["clicks_14"] = f["active_14"] * 8.0
    f["clicks_3"] = min(3, active7) * 8.0 if profile.days_since_active < 3 else 0.0
    f["clicks_30"] = active7 * 30 / 7 * 8.0
    if rate is not None:  # people who are usually on time also hand in a few days early (OULAD: about +4 days)
        f["prior_mean_days_early"] = round((rate - 0.5) * 8, 2)
        f["last_prior_days_early"] = f["prior_mean_days_early"]
    return f


APP_FIXED = ("is_tma",)  # constant for app tasks (see features_from_app), so never a "reason" for one task


class CompletionModel:
    def __init__(self, models_dir: Path | str = MODELS_DIR):
        self.dir = Path(models_dir)

    @property
    def available(self) -> bool:
        card = self.dir / "model_card.json"
        return card.exists() and (self.dir / json.loads(card.read_text())["xgboost_file"]).exists()

    @cached_property
    def card(self) -> dict:
        return json.loads((self.dir / "model_card.json").read_text())

    @cached_property
    def defaults(self) -> dict:
        p = self.dir / "completion_defaults.json"
        return json.loads(p.read_text()) if p.exists() else {"median": {}, "categories": {}}

    @cached_property
    def booster(self):
        import xgboost as xgb

        m = xgb.XGBClassifier()
        m.load_model(str(self.dir / self.card["xgboost_file"]))
        return m

    @property
    def features(self) -> list[str]:
        return self.card["features"]

    @property
    def threshold(self) -> float:
        return float(self.card["threshold"])

    def frame(self, rows: list[dict]) -> pd.DataFrame:
        med, cats = self.defaults.get("median", {}), self.defaults.get("categories", {})
        data = []
        for r in rows:
            row = {}
            for c in self.features:
                if c in r:  # given: None/NaN means "unknown" (e.g. no history yet), which the model handles itself
                    v = r[c]
                    if v is None or (isinstance(v, float) and math.isnan(v)):
                        v = np.nan if c not in cats else self.defaults.get("mode", {}).get(c)
                else:  # not given: use the typical training value
                    v = med.get(c, np.nan) if c not in cats else self.defaults.get("mode", {}).get(c)
                row[c] = v
            data.append(row)
        df = pd.DataFrame(data, columns=self.features)
        for c, levels in cats.items():
            if c in df:
                df[c] = pd.Categorical(df[c].astype(str), categories=levels)
        for c in df.columns:
            if c not in cats:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
        return df

    def predict(self, rows: list[dict]) -> np.ndarray:
        return self.booster.predict_proba(self.frame(rows))[:, 1]

    def explain(self, row: dict, top: int = 5, skip: tuple[str, ...] = ()) -> list[dict]:
        import xgboost as xgb

        X = self.frame([row])  # noqa: N806
        contrib = self.booster.get_booster().predict(xgb.DMatrix(X, enable_categorical=True), pred_contribs=True)[0]
        pairs = [p for p in zip(self.features, contrib[:-1], strict=True) if p[0] not in skip]
        pairs = sorted(pairs, key=lambda x: -abs(x[1]))[:top]
        out = []
        for name, c in pairs:
            v = X.iloc[0][name]
            out.append(
                {
                    "feature": name,
                    "label": LABELS.get(name, name),
                    "value": None
                    if pd.isna(v)
                    else (str(v) if name in self.defaults.get("categories", {}) else float(v)),
                    "effect": round(float(c), 4),
                    "direction": "raises" if c > 0 else "lowers",
                }
            )
        return out
