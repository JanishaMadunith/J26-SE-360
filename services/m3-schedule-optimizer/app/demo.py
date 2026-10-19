"""Model and demo endpoints for Module 3 (used by the /demo page and visible in /docs).

POST /v1/completion/predict   chance a task is finished on time + the top reasons
GET  /v1/completion/examples  real held-out OULAD cases (2014J) with what really happened
POST /v1/tasktime             best part of the day for task titles (MS-LaTTE model)
GET  /v1/demo/context         what the demo can use: ATUS personas, PMData wellness days
POST /v1/demo/plan            end-to-end: tasks + profile + persona + wellness day -> scheduled plan
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from m3.completion_model import APP_FIXED, BehaviourProfile, features_from_app
from m3.context_datasets import forecast_from_days
from m3.tasktime import PARTS, predict_parts
from wellness_contracts import ScheduleRequest, TaskCategory

from .predictor import Predictor

TZ = ZoneInfo("Asia/Colombo")


class ProfileIn(BaseModel):
    on_time_rate: float | None = Field(0.8, ge=0, le=1, description="Share of past tasks finished on time")
    tasks_done_before: int = Field(8, ge=0)
    days_since_active: float = Field(1, ge=0)
    active_days_7: int = Field(5, ge=0, le=7)
    unfinished_earlier_tasks: int = Field(0, ge=0)
    overdue_tasks: int = Field(0, ge=0)
    upcoming_deadlines_7d: int = Field(1, ge=0)

    def to_profile(self) -> BehaviourProfile:
        return BehaviourProfile(
            on_time_rate=self.on_time_rate,
            same_type_on_time_rate=self.on_time_rate,
            tasks_done_before=self.tasks_done_before,
            days_since_active=self.days_since_active,
            active_days_7=self.active_days_7,
            unfinished_earlier_tasks=self.unfinished_earlier_tasks,
            overdue_tasks=self.overdue_tasks,
            upcoming_deadlines_7d=self.upcoming_deadlines_7d,
        )


class TaskIn(BaseModel):
    title: str = "Write project report"
    category: TaskCategory = TaskCategory.study
    estimated_minutes: int = Field(120, ge=15, le=600)
    priority: int = Field(4, ge=1, le=5)
    due_in_days: float = Field(3, ge=0.25, le=14)


class CompletionRequest(BaseModel):
    features: dict | None = Field(None, description="Raw model features (OULAD names); others use training medians")
    profile: ProfileIn | None = None
    task: TaskIn | None = None


class PlanRequest(BaseModel):
    persona: Literal["working_adult", "university_student", "free"] = "university_student"
    wellness_day: Literal["good", "typical", "hard"] = "typical"
    profile: ProfileIn = ProfileIn()
    tasks: list[TaskIn] = Field(min_length=1, max_length=12)
    days: int = Field(3, ge=1, le=5)


@lru_cache(maxsize=4)
def _json(path: str):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def router(pred: Predictor) -> APIRouter:
    r = APIRouter(tags=["models & demo"])
    md = pred.models_dir

    def need_completion():
        if not pred.has_completion:
            raise HTTPException(
                503, "Completion model not found in models/ - see services/m3-schedule-optimizer/README"
            )

    @r.post("/v1/completion/predict")
    def completion_predict(req: CompletionRequest) -> dict:
        need_completion()
        if req.features is not None:
            row = dict(req.features)
        else:
            t = req.task or TaskIn()
            row = features_from_app(
                (req.profile or ProfileIn()).to_profile(), t.due_in_days, t.estimated_minutes, t.priority
            )
        p = float(pred.completion.predict([row])[0])
        thr = pred.completion.threshold
        return {
            "probability_on_time": round(p, 4),
            "prediction": "on time" if p >= thr else "at risk",
            "threshold": round(thr, 3),
            "reasons": pred.completion.explain(row, top=5, skip=() if req.features is not None else APP_FIXED),
            "model": "XGBoost trained on OULAD (663,420 cases); test accuracy 83.4%, ROC-AUC 0.916",
        }

    @r.get("/v1/completion/examples")
    def completion_examples() -> list[dict]:
        ex = _json(str(md / "demo_examples.json"))
        if ex is None:
            raise HTTPException(503, "models/demo_examples.json missing - run: python -m m3.build_demo_assets")
        return ex

    @r.post("/v1/tasktime")
    def tasktime(titles: list[str]) -> list[dict]:
        out = predict_parts(titles, md)
        if out is None:
            raise HTTPException(503, "Task time-of-day model missing - run: python -m m3.tasktime")
        return [{"title": t, "parts": p, "best": max(p, key=p.get)} for t, p in zip(titles, out, strict=True)]

    @r.get("/v1/demo/context")
    def context() -> dict:
        atus = _json(str(md / "atus_day_profiles.json")) or {"personas": {}}
        pm = _json(str(md / "pmdata_wellness_days.json")) or []
        return {
            "personas": {k: v["default_busy_blocks"] for k, v in atus["personas"].items()},
            "pmdata_days": len(pm),
            "pmdata_participants": len({d["participant"] for d in pm}),
            "models": pred.status(),
        }

    def wellness_days(kind: str, n: int) -> list[dict]:
        pm = _json(str(md / "pmdata_wellness_days.json"))
        if not pm:
            return [{"energy": 0.6, "focus": 0.6, "calm": 0.6, "participant": "neutral", "date": "-"}] * n
        q = {"good": 0.9, "typical": 0.5, "hard": 0.1}[kind]
        ordered = sorted(pm, key=lambda d: d["energy"])
        first = ordered[int(q * (len(ordered) - 1))]
        same = [d for d in pm if d["participant"] == first["participant"] and d["date"] >= first["date"]]
        same.sort(key=lambda d: d["date"])
        return (same + [same[-1]] * n)[:n]

    @r.post("/v1/demo/plan")
    def plan(req: PlanRequest) -> dict:
        now = datetime.now(TZ)
        start = (now + timedelta(days=1)).replace(hour=7, minute=0, second=0, microsecond=0)
        while start.weekday() >= 5:  # start on a weekday so the persona's work/class hours apply
            start += timedelta(days=1)
        end = start + timedelta(days=req.days)
        atus = _json(str(md / "atus_day_profiles.json")) or {"personas": {}}
        blocks = [] if req.persona == "free" else atus["personas"].get(req.persona, {}).get("default_busy_blocks", [])
        busy = []
        for d in range(req.days):
            day = start + timedelta(days=d)
            if day.weekday() >= 5:
                continue
            for a, b in blocks:
                ha, ma = map(int, a.split(":"))
                hb, mb = map(int, b.split(":"))
                s = day.replace(hour=ha, minute=ma)
                e = day.replace(hour=hb, minute=mb) if (hb, mb) != (0, 0) else day.replace(hour=23, minute=59)
                busy.append({"start": s.isoformat(), "end": e.isoformat(), "label": "work/class (ATUS)"})
        days = wellness_days(req.wellness_day, req.days + 1)
        uid = str(uuid.uuid4())
        tasks = []
        for t in req.tasks:
            tasks.append(
                {
                    "task_id": str(uuid.uuid4()),
                    "title": t.title,
                    "category": t.category.value,
                    "estimated_minutes": t.estimated_minutes,
                    "priority": t.priority,
                    "deadline": (start + timedelta(days=t.due_in_days)).isoformat(),
                }
            )
        sreq = ScheduleRequest.model_validate(
            {
                "user_id": uid,
                "as_of": start.isoformat(),
                "timezone": "Asia/Colombo",
                "horizon_start": start.isoformat(),
                "horizon_end": end.isoformat(),
                "tasks": tasks,
                "busy": busy,
                "wellness": forecast_from_days(days, uid, start, hours=24 * req.days),
            }
        )
        prof = req.profile.to_profile()
        profiles = {t.task_id: prof for t in sreq.tasks}
        priors = pred.priors(sreq, profiles)
        res = pred.predict(sreq, profiles)
        out_tasks = []
        for t, tin in zip(sreq.tasks, req.tasks, strict=True):
            pr = priors.get(t.task_id)
            row = features_from_app(prof, tin.due_in_days, t.estimated_minutes, t.priority)
            out_tasks.append(
                {
                    "task_id": str(t.task_id),
                    "title": t.title,
                    "category": t.category.value,
                    "estimated_minutes": t.estimated_minutes,
                    "priority": t.priority,
                    "deadline": t.deadline.isoformat(),
                    "p_on_time_model": None if pr is None or pr.p_on_time is None else round(pr.p_on_time, 3),
                    "parts": None if pr is None else pr.parts,
                    "best_part": None if pr is None or not pr.parts else max(pr.parts, key=pr.parts.get),
                    "reasons": pred.completion.explain(row, top=3, skip=APP_FIXED) if pred.has_completion else [],
                    "sessions": [
                        {"start": i.start.isoformat(), "end": i.end.isoformat(), "p": i.completion_probability}
                        for i in res.items
                        if i.task_id == t.task_id
                    ],
                    "at_risk": t.task_id in res.at_risk_task_ids,
                    "unscheduled": next((u.reason for u in res.unscheduled if u.task_id == t.task_id), None),
                }
            )
        return {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "busy": busy,
            "wellness": {
                "source": f"PMData {days[0]['participant']} from {days[0]['date']} ({req.wellness_day} day)",
                "hourly": [
                    {"start": s.start.isoformat(), "energy": s.energy, "focus": s.focus} for s in sreq.wellness.slots
                ],
            },
            "tasks": out_tasks,
            "overload_predicted": res.overload_predicted,
            "model": pred.status(),
            "parts": PARTS,
        }

    return r
