"""Model and demo endpoints for Module 3 (used by the /demo page and visible in /docs).

POST /v1/completion/predict   chance a task is finished on time + the top reasons
GET  /v1/completion/examples  real held-out OULAD cases (2014J) with what really happened
POST /v1/tasktime             best part of the day for task titles (MS-LaTTE model)
GET  /v1/demo/context         what the demo can use: ATUS personas, PMData wellness days
POST /v1/demo/plan            end-to-end: tasks + profile + persona + wellness day -> scheduled plan
POST /v1/demo/replan          adaptive re-planning after a missed task or a new urgent task (minimal moves)
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
from pydantic import BaseModel, Field, field_validator

from m3.completion_model import APP_FIXED, BehaviourProfile, features_from_app
from m3.context_datasets import SLEEP_H_PER_DEADLINE, STRESS_PER_DEADLINE, apply_deadline_load, forecast_from_days
from m3.scheduling.heuristic import build_grid
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
    usual_days_early: float | None = Field(None, ge=-7, le=7, description="Usually finishes this many days early")

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
            usual_days_early=self.usual_days_early,
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


class PlanTask(BaseModel):
    title: str = Field("Write project report", min_length=1, max_length=200)
    category: TaskCategory = TaskCategory.study
    estimated_minutes: int = Field(60, ge=15, le=600)
    priority: int = Field(3, ge=1, le=5)
    due_day: int | None = Field(
        2,
        ge=-1,
        le=13,
        description="0 = today, 1 = tomorrow, ... (due 23:00 that day); -1 = already passed; null = none",
    )
    fixed: str | None = Field(
        None,
        pattern=r"^\s*\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}\s*$",
        description="Fixed event time, e.g. '08:00-13:00' (a lecture); on the due day (today if no due day)",
    )


class ManualSlot(BaseModel):
    start: datetime
    end: datetime | None = None
    energy: float = Field(ge=0, le=1)
    focus: float = Field(ge=0, le=1)
    mental_fatigue: float = Field(0.3, ge=0, le=1)
    burnout_risk: float = Field(0.2, ge=0, le=1)


class ManualForecast(BaseModel):
    """What Module 2 sends (WellnessForecast). user_id, generated_at and model may be left out here."""

    slot_minutes: int = Field(60, ge=15, le=240)
    slots: list[ManualSlot] = Field(min_length=1, max_length=24 * 6 * 4)
    sleep_hours_last_night: float | None = Field(None, ge=0, le=16)


class ManualInput(BaseModel):
    """Manual mode of the demo: type Module 2's forecast and the busy hours yourself instead of PMData / ATUS."""

    wellness: ManualForecast | None = Field(None, description="A Module 2 WellnessForecast; null = PMData stand-in")
    busy: list[str] | None = Field(
        None,
        description="Busy hours every planned day, e.g. ['09:00-15:30']; null = the ATUS persona",
    )
    deadline_effect: bool = Field(
        False,
        description="Also lower wellness for deadlines (StudentLife); off when Module 2 already includes workload",
    )

    @field_validator("busy")
    @classmethod
    def _check_busy(cls, v):
        import re

        for b in v or []:
            if not re.fullmatch(r"\s*\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}\s*", b):
                raise ValueError(f"busy block '{b}' must look like 09:00-15:30")
        return v


class PlanRequest(BaseModel):
    persona: Literal["working_adult", "university_student", "free"] = "university_student"
    wellness_day: Literal["good", "typical", "hard"] = "typical"
    profile: ProfileIn = ProfileIn()
    tasks: list[PlanTask] = Field(default=[], max_length=15)
    days: int = Field(3, ge=1, le=5)
    day_end_hour: int = Field(23, ge=18, le=24, description="Latest hour for flexible tasks")
    manual: ManualInput | None = Field(None, description="Manual mode: your own Module 2 forecast and busy hours")


class Session(BaseModel):
    task_key: str
    start: datetime
    end: datetime
    p: float = 0.5


class Change(BaseModel):
    kind: Literal["missed", "add"]
    task_key: str | None = Field(None, description="missed: the task that was not done (its first session passed)")
    session_start: datetime | None = Field(
        None,
        description="missed: which of the task's sessions was missed (clicked on the timeline); default: the first",
    )
    new_task: PlanTask | None = Field(None, description="add: the new (urgent) task")


class ReplanRequest(BaseModel):
    plan: PlanRequest
    previous: list[Session]
    change: Change


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
            "plan_start": day_one().isoformat(),
            "models": pred.status(),
        }

    @r.get("/v1/demo/prepared")
    def prepared() -> dict:
        """Preview of the prepared (model-ready) data built from each dataset (models/prepared_preview.json)."""
        pv = _json(str(md / "prepared_preview.json"))
        if pv is None:
            raise HTTPException(503, "models/prepared_preview.json missing - run: python -m m3.build_prepared_preview")
        return pv

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

    def day_one() -> datetime:
        """'Today' of the demo: the next weekday at 07:00 (so the persona's work/class hours apply)."""
        start = (datetime.now(TZ) + timedelta(days=1)).replace(hour=7, minute=0, second=0, microsecond=0)
        while start.weekday() >= 5:
            start += timedelta(days=1)
        return start

    def busy_blocks(persona: str, start: datetime, n_days: int) -> list[dict]:
        atus = _json(str(md / "atus_day_profiles.json")) or {"personas": {}}
        blocks = [] if persona == "free" else atus["personas"].get(persona, {}).get("default_busy_blocks", [])
        busy = []
        for d in range(n_days):
            day = start + timedelta(days=d)
            if day.weekday() >= 5:
                continue
            for a, b in blocks:
                ha, ma = map(int, a.split(":"))
                hb, mb = map(int, b.split(":"))
                s0 = day.replace(hour=ha, minute=ma)
                e0 = day.replace(hour=hb, minute=mb) if (hb, mb) != (0, 0) else day.replace(hour=23, minute=59)
                busy.append({"start": s0.isoformat(), "end": e0.isoformat(), "label": "work/class (ATUS)"})
        return busy

    def manual_busy(blocks: list[str], start: datetime, n_days: int) -> list[dict]:
        busy = []
        for d in range(n_days):
            day = start + timedelta(days=d)
            for blk in blocks:
                a, b = (x.strip() for x in blk.split("-"))
                s0 = day.replace(hour=int(a.split(":")[0]), minute=int(a.split(":")[1]))
                e0 = day.replace(hour=min(23, int(b.split(":")[0])), minute=int(b.split(":")[1]))
                if b.startswith("24") or b == "00:00":
                    e0 = day.replace(hour=23, minute=59)
                if e0 > s0:
                    busy.append({"start": s0.isoformat(), "end": e0.isoformat(), "label": "busy (manual)"})
        return busy

    def manual_forecast(mf: ManualForecast, uid: str, start: datetime, hours: int, loads: list[int]) -> dict:
        """Module 2's forecast as typed or pasted: each planned hour takes the slot that covers it; if the dates do
        not match the plan, the slot with the same time of day; hours not given are neutral. With deadline_effect,
        the StudentLife effects are applied on top (as for the PMData stand-in)."""
        step = timedelta(minutes=mf.slot_minutes)
        spans = [(s.start, s.end or s.start + step, s) for s in mf.slots]

        def tod(t: datetime) -> int:
            t = t.astimezone(TZ)
            return t.hour * 60 + t.minute

        def find(t: datetime):
            for a, b, s in spans:
                if a <= t < b:
                    return s
            m = tod(t)
            for a, b, s in spans:
                lo, ln = tod(a), int((b - a).total_seconds() // 60)
                if 0 <= (m - lo) % 1440 < ln:
                    return s
            return None

        slots = []
        start = start.replace(minute=0, second=0, microsecond=0)
        for h in range(hours):
            t = start + timedelta(hours=h)
            s = find(t)
            v = (
                {
                    "energy": s.energy,
                    "focus": s.focus,
                    "mental_fatigue": s.mental_fatigue,
                    "burnout_risk": s.burnout_risk,
                }
                if s
                else {"energy": 0.5, "focus": 0.5, "mental_fatigue": 0.3, "burnout_risk": 0.2}
            )
            load = loads[min(len(loads) - 1, (t.date() - start.date()).days)] if loads else 0
            if load:
                stress, sleep = STRESS_PER_DEADLINE * load / 4, -SLEEP_H_PER_DEADLINE * load / 7
                v = {
                    "energy": max(0.05, v["energy"] - sleep),
                    "focus": max(0.05, v["focus"] - stress),
                    "mental_fatigue": min(1.0, v["mental_fatigue"] + sleep),
                    "burnout_risk": min(1.0, v["burnout_risk"] + stress),
                }
            slots.append(
                {
                    "start": t.isoformat(),
                    "end": (t + timedelta(hours=1)).isoformat(),
                    **{k: round(x, 3) for k, x in v.items()},
                }
            )
        return {
            "user_id": uid,
            "generated_at": start.isoformat(),
            "slot_minutes": 60,
            "slots": slots,
            "sleep_hours_last_night": mf.sleep_hours_last_night,
            "model": {"name": "m2-manual-demo", "version": "1.0"},
        }

    def key_id(key: str) -> uuid.UUID:
        return uuid.uuid5(uuid.NAMESPACE_URL, f"m3-demo/{key}")

    def deadline_of(t: PlanTask, start: datetime) -> datetime | None:
        if t.due_day is None:
            return None
        day = start.replace(hour=23, minute=0) + timedelta(days=t.due_day)
        return day if t.due_day >= 0 else start - timedelta(hours=1)  # -1: already passed

    def run(req: PlanRequest, as_of: datetime | None = None, locked: list[Session] | None = None,
            keys: list[str] | None = None, bumped: dict | None = None) -> dict:  # fmt: skip
        """Build the ScheduleRequest, run models + scheduler, and shape the answer for the demo page."""
        start = day_one()
        as_of = as_of or start
        end = start + timedelta(days=req.days)
        keys = keys or [f"t{i}" for i in range(len(req.tasks))]
        locked = locked or []
        locked_keys = {s.task_key for s in locked}
        man = req.manual or ManualInput()
        busy = (
            manual_busy(man.busy, start, req.days)
            if man.busy is not None
            else busy_blocks(req.persona, start, req.days)
        )
        days = wellness_days(req.wellness_day, req.days + 1)
        # StudentLife: each deadline in the next 3 days raises stress and cuts sleep -> lower that day's wellness
        dls = [deadline_of(t, start) for t in req.tasks]
        loads = []
        for i in range(len(days)):
            d0 = start.replace(hour=0) + timedelta(days=i)
            loads.append(sum(1 for dl in dls if dl is not None and d0 <= dl < d0 + timedelta(days=3)))
        manual_w = man.wellness is not None
        use_load = man.deadline_effect if manual_w else True  # manual Module 2 input: StudentLife only if ticked
        if not use_load:
            loads = [0] * len(loads)
        days = [apply_deadline_load(d, n) for d, n in zip(days, loads, strict=True)]
        uid = str(uuid.uuid4())
        wellness = (
            manual_forecast(man.wellness, uid, start, 24 * req.days, loads)
            if manual_w
            else forecast_from_days(days, uid, start, hours=24 * req.days)
        )
        tasks = []
        for k, t in zip(keys, req.tasks, strict=True):
            base = {
                "title": t.title,
                "category": t.category.value,
                "estimated_minutes": t.estimated_minutes,
                "priority": t.priority,
                "times_rescheduled": (bumped or {}).get(k, 0),
            }
            dl = deadline_of(t, start)
            if dl is not None:
                base["deadline"] = dl.isoformat()
            if t.fixed:  # a fixed event (lecture, meeting): stays exactly at its time
                a, b = (x.strip() for x in t.fixed.split("-"))
                day = start + timedelta(days=max(0, t.due_day or 0))
                fs = day.replace(hour=int(a.split(":")[0]), minute=int(a.split(":")[1]))
                fe = day.replace(hour=int(b.split(":")[0]), minute=int(b.split(":")[1]))
                if fe > fs:
                    tasks.append(
                        {**base, "task_id": str(key_id(k)), "fixed_start": fs.isoformat(), "fixed_end": fe.isoformat()}
                    )
                    continue
            if k in locked_keys:  # keep its existing sessions exactly where they are (minimal moves)
                for s in (s for s in locked if s.task_key == k):
                    tasks.append(
                        {
                            **base,
                            "task_id": str(key_id(k)),
                            "fixed_start": s.start.isoformat(),
                            "fixed_end": s.end.isoformat(),
                        }
                    )
            else:
                tasks.append({**base, "task_id": str(key_id(k))})
        sreq = ScheduleRequest.model_validate(
            {
                "user_id": uid,
                "as_of": as_of.isoformat(),
                "timezone": "Asia/Colombo",
                "horizon_start": start.isoformat(),
                "horizon_end": end.isoformat(),
                "day_start_hour": 6,
                "day_end_hour": req.day_end_hour,
                "tasks": tasks,
                "busy": busy,
                "wellness": wellness,
            }
        )
        prof = req.profile.to_profile()
        prof.upcoming_deadlines_7d = max(  # other deadlines in the next 7 days = workload signal for the model
            prof.upcoming_deadlines_7d,
            sum(1 for dl in dls if dl is not None and as_of <= dl < as_of + timedelta(days=7)) - 1,
        )
        profiles = {t.task_id: prof for t in sreq.tasks}
        priors = pred.priors(sreq, profiles)
        res = pred.predict(sreq, profiles)
        prev_p = {(s.task_key, s.start): s.p for s in locked}
        meals = [{"start": a.isoformat(), "end": b.isoformat()} for a, b in build_grid(sreq).meals]
        out_tasks = []
        for k, t in zip(keys, req.tasks, strict=True):
            tid = key_id(k)
            pr = priors.get(tid)
            dl = deadline_of(t, start)
            ref = start if k in locked_keys else as_of  # kept tasks: rated as when they were planned
            days_left = 14.0 if dl is None else max(0.0, (dl - ref).total_seconds() / 86400)
            row = features_from_app(prof, days_left, t.estimated_minutes, t.priority)
            p_model = pr.p_on_time if pr is not None and pr.p_on_time is not None else None
            if p_model is None and pred.has_completion:  # locked tasks are fixed in the request: rate them anyway
                p_model = float(pred.completion.predict([row])[0])
            parts = pr.parts if pr is not None else (predict_parts([t.title], md, [t.category.value]) or [None])[0]
            sessions = [
                {
                    "start": i.start.isoformat(),
                    "end": i.end.isoformat(),
                    "p": prev_p.get((k, i.start), i.completion_probability),
                    "kept": k in locked_keys,
                }
                for i in res.items
                if i.task_id == tid
            ]
            unsched = next((u.reason for u in res.unscheduled if u.task_id == tid), None)
            out_tasks.append(
                {
                    "task_key": k,
                    "title": t.title,
                    "category": t.category.value,
                    "estimated_minutes": t.estimated_minutes,
                    "priority": t.priority,
                    "due_day": t.due_day,
                    "deadline": None if dl is None else dl.isoformat(),
                    "p_on_time_model": None if p_model is None or t.fixed else round(p_model, 3),
                    "parts": parts,
                    "best_part": None if not parts else max(parts, key=parts.get),
                    "reasons": pred.completion.explain(row, top=3, skip=APP_FIXED)
                    if pred.has_completion and not t.fixed
                    else [],
                    "fixed": t.fixed,
                    "sessions": sessions,
                    "at_risk": not t.fixed and (unsched is not None or any(x["p"] < 0.5 for x in sessions)),
                    "unscheduled": unsched,
                }
            )
        planned_min = sum(t.estimated_minutes for t in req.tasks)
        return {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "as_of": as_of.isoformat(),
            "busy": busy,
            "meals": meals,
            "wellness": {
                "source": (
                    f"Manual Module 2 input ({len(man.wellness.slots)} slot(s) given)"
                    if manual_w
                    else f"PMData {days[0]['participant']} from {days[0]['date']} ({req.wellness_day} day)"
                )
                + (f"; lowered for deadline load {loads[: req.days]} (StudentLife)" if any(loads) else ""),
                "manual": manual_w,
                "busy_source": "manual" if man.busy is not None else req.persona,
                "hourly": [
                    {
                        "start": s.start.isoformat(),
                        "energy": s.energy,
                        "focus": s.focus,
                        "mental_fatigue": s.mental_fatigue,
                        "burnout_risk": s.burnout_risk,
                    }
                    for s in sreq.wellness.slots
                ],
            },
            "tasks": out_tasks,
            "summary": {
                "tasks": len(req.tasks),
                "planned_hours": round(planned_min / 60, 1),
                "at_risk": sum(t["at_risk"] for t in out_tasks),
                "unscheduled": sum(t["unscheduled"] is not None for t in out_tasks),
            },
            "overload_predicted": res.overload_predicted,
            "model": pred.status(),
            "parts": PARTS,
            # the real Module 3 contract this plan was made with: what the Core API sends and what Module 3 returns
            "contract": {
                "endpoint": "POST /v1/schedule",
                "task_titles": {str(t.task_id): t.title for t in sreq.tasks},
                "request": sreq.model_dump(mode="json"),
                "response": res.model_dump(mode="json"),
            },
        }

    @r.post("/v1/demo/plan")
    def plan(req: PlanRequest) -> dict:
        return run(req)

    @r.post("/v1/demo/replan")
    def replan(req: ReplanRequest) -> dict:
        """Adaptive re-planning with minimal moves.

        missed: the task's first planned session has passed undone -> 'now' moves to the end of that session, the
                task is re-placed (counted as rescheduled), every other future session stays where it was.
        add:    a new task arrives now -> existing future sessions stay; the new task goes into the free time.
        If the changed task cannot fit before its deadline while the rest stay fixed, the whole remaining plan is
        rebuilt and the moved sessions are reported."""
        plan_req = req.plan.model_copy(deep=True)
        keys = [f"t{i}" for i in range(len(plan_req.tasks))]
        start = day_one()
        prev = sorted(req.previous, key=lambda s: s.start)
        bumped: dict = {}
        mine: list = []
        if req.change.kind == "missed":
            k = req.change.task_key
            mine = [s for s in prev if s.task_key == k]
            if not mine:
                raise HTTPException(422, f"task {k} has no planned session to miss")
            if req.change.session_start is not None:  # the session the user clicked; the task is re-placed after it
                mine = [s for s in mine if s.start == req.change.session_start] or mine
            now = mine[0].end
            bumped[k] = 1
            changed = k
        else:
            if req.change.new_task is None:
                raise HTTPException(422, "add needs new_task")
            plan_req.tasks.append(req.change.new_task)
            changed = f"t{len(keys)}"
            keys.append(changed)
            now = start  # the new task arrives at the start of today
        # every other task keeps its sessions (past ones are history, future ones stay put = minimal moves)
        locked = [s for s in prev if s.task_key != changed]
        out = run(plan_req, as_of=now, locked=locked, keys=keys, bumped=bumped)
        target = next(t for t in out["tasks"] if t["task_key"] == changed)
        mode = "minimal"
        if target["unscheduled"]:  # it cannot fit while everything else stays: rebuild the rest of the plan
            out = run(plan_req, as_of=now, locked=[], keys=keys, bumped=bumped)
            mode = "full"
        old = {(s.task_key, s.start.isoformat()) for s in prev if s.task_key != changed}
        new = {(t["task_key"], x["start"]) for t in out["tasks"] if t["task_key"] != changed for x in t["sessions"]}
        moved = sorted({k for k, _ in old - new})
        out["replan"] = {
            "change": req.change.kind,
            "changed_task": changed,
            "now": now.isoformat(),
            "missed_slot": (
                {"start": mine[0].start.isoformat(), "end": mine[0].end.isoformat()}
                if req.change.kind == "missed"
                else None
            ),
            "mode": mode,
            "kept_sessions": len(old & new),
            "moved_tasks": moved,
        }
        return out

    return r
