"""Module 3 predictor: trained models + schedule optimiser.

v0.2.0
  1. Completion model (XGBoost, trained on OULAD, src/m3/train_oulad.py): chance each task is finished on time,
     from the user's behaviour (task-event log) and the task (time left, size, priority).
  2. Task time-of-day model (TF-IDF + logistic regression, trained on MS-LaTTE, src/m3/tasktime.py): which part
     of the day people usually do this kind of task.
  3. Energy-aware scheduler (src/m3/scheduling/heuristic.py): places each task in the slot with the best
     adjusted chance, using Module 2's hourly wellness forecast.

If a model file is missing (e.g. a fresh clone without models/), the service falls back to the transparent
baseline and says so in /health. NEXT: OR-Tools CP-SAT optimiser behind the same interface
(src/m3/scheduling/cpsat.py), with this scheduler kept as the evaluation baseline.
"""

from __future__ import annotations

import logging
from pathlib import Path

from m3.completion_model import BehaviourProfile, CompletionModel, features_from_app
from m3.scheduling.heuristic import MODEL as BASELINE_MODEL
from m3.scheduling.heuristic import TaskPrior, schedule
from m3.tasktime import predict_parts
from wellness_contracts import ModelInfo, ScheduleRequest, ScheduleResult, UserHistory

log = logging.getLogger(__name__)
ML_MODEL = ModelInfo(name="m3-ml-energy-aware", version="0.2.0")


def profile_from_history(history: UserHistory, category) -> BehaviourProfile:
    done = history.tasks_completed_30d + history.tasks_missed_30d
    return BehaviourProfile(
        on_time_rate=history.on_time_rate_30d,
        same_type_on_time_rate=history.on_time_rate_by_category.get(category),
        tasks_done_before=done,
        overdue_tasks=history.tasks_missed_30d,
    )


class Predictor:
    def __init__(self, models_dir: str | Path | None = None):
        from m3.config import MODELS_DIR

        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.completion = CompletionModel(self.models_dir)
        self.has_completion = self.completion.available
        self.has_tasktime = predict_parts(["test"], self.models_dir) is not None
        self.info: ModelInfo = ML_MODEL if self.has_completion else BASELINE_MODEL
        if not self.has_completion:
            log.warning("completion model not found in %s - using the baseline scheduler", self.models_dir)

    def status(self) -> dict:
        return {
            **self.info.model_dump(),
            "completion_model": "xgboost-oulad" if self.has_completion else "missing (baseline used)",
            "tasktime_model": "tfidf-logreg-mslatte" if self.has_tasktime else "missing",
        }

    def priors(self, req: ScheduleRequest, profiles: dict | None = None) -> dict:
        """Trained-model view of each flexible task: chance on time + time-of-day preference."""
        flexible = [t for t in req.tasks if not (t.fixed_start and t.fixed_end)]
        if not flexible:
            return {}
        parts = (
            predict_parts([t.title for t in flexible], self.models_dir, [t.category.value for t in flexible])
            if self.has_tasktime
            else None
        )
        probs = [None] * len(flexible)
        if self.has_completion:
            rows = []
            for t in flexible:
                days = 14.0 if t.deadline is None else (t.deadline - req.as_of).total_seconds() / 86400
                prof = (profiles or {}).get(t.task_id) or profile_from_history(req.history, t.category)
                rows.append(features_from_app(prof, days, t.estimated_minutes, t.priority))
            probs = [float(p) for p in self.completion.predict(rows)]
        return {
            t.task_id: TaskPrior(p_on_time=probs[i], parts=None if parts is None else parts[i])
            for i, t in enumerate(flexible)
        }

    def predict(self, req: ScheduleRequest, profiles: dict | None = None) -> ScheduleResult:
        if not (self.has_completion or self.has_tasktime):
            return schedule(req)
        return schedule(req, self.priors(req, profiles), model=self.info)
