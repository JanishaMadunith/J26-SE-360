"""Module 3 serving baseline — energy-aware greedy scheduler (pure functions, no I/O).

Algorithm
  1. GRID      Split the horizon into 15-minute cells. A cell is available if it lies inside the user's local
               day window [day_start_hour, day_end_hour), after `as_of`, does not overlap a busy block
               or a fixed task, and is not a protected meal break (MEAL_BREAKS, moved later if a lecture covers it — so Module 4 always finds a
               free gap to place meals). Every cell carries the Module 2 wellness values for its time
               (neutral if absent).
  2. FIXED     Fixed tasks (lectures, meetings) are kept exactly where they are.
  3. ORDER     Flexible tasks are ordered by urgency = 0.6 / (1 + hours_to_deadline / 24) + 0.4 * priority / 5.
               Tasks longer than 120 min are split into focused sessions of <= 120 min.
  4. PLACE     Each session goes to the contiguous free block (ending before the deadline) with the highest
               expected completion probability (transparent logistic model below). A soft penalty avoids
               building more than 3 hours of back-to-back flexible work.
  5. FLAGS     at-risk = a flexible session scheduled with probability < 0.5 (fixed events are attended, not rated); overload = earliest-deadline-first demand exceeds the
               free minutes before some deadline, or the mean forecast burnout risk > 0.7.

Completion chance per slot: when the service passes a TaskPrior (trained completion model on OULAD + task
time-of-day model on MS-LaTTE), the task-level probability from the trained model is adjusted for the slot's
wellness, length, hour and time-of-day fit. Without priors, a transparent hand-set logistic baseline is used
(COEF) — kept as the comparison baseline.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from uuid import UUID
from zoneinfo import ZoneInfo

from wellness_contracts import (
    ModelInfo,
    SchedulableTask,
    ScheduledItem,
    ScheduleRequest,
    ScheduleResult,
    TaskCategory,
    UnscheduledTask,
    UserHistory,
    WellnessForecast,
    utc_now,
)

MODEL = ModelInfo(name="m3-greedy-energy-aware", version="0.1.0")
CELL_MIN = 15
CELL = timedelta(minutes=CELL_MIN)
MAX_SESSION_MIN = 120
MAX_CONTINUOUS_MIN = 180
RISK_THRESHOLD = 0.5
BURNOUT_OVERLOAD = 0.7
NO_DEADLINE_SLACK_H = 72.0
# Local-time meal breaks kept free of flexible tasks: (hour, minute, length_min). They match Module 4's usual
# meal times, so meals always have a free slot. Fixed tasks may still overlap them (the user chose that time).
MEAL_BREAKS = ((7, 0, 30), (12, 30, 30), (19, 30, 30))
MEAL_SHIFT_MAX_MIN = 120  # a meal break covered by a lecture/meeting moves up to 2 h later

# How much a category depends on focus (cognitive demand) and on physical energy.
FOCUS_DEMAND = {
    TaskCategory.study: 1.0,
    TaskCategory.work: 0.9,
    TaskCategory.exercise: 0.2,
    TaskCategory.chores: 0.2,
    TaskCategory.personal: 0.3,
    TaskCategory.social: 0.2,
    TaskCategory.other: 0.5,
}
ENERGY_DEMAND = {TaskCategory.exercise: 1.6, TaskCategory.personal: 0.4, TaskCategory.social: 0.5}  # default 1.0

COEF = {
    "intercept": 0.6,
    "log_slack_h": 0.35,  # more time to spare before the deadline -> more likely on time
    "energy": 1.2,  # x energy demand of the category, centred at 0.5
    "focus": 1.6,  # x focus demand of the category, centred at 0.5
    "fatigue": -1.0,  # centred at 0.5
    "duration_h": -0.35,  # longer sessions are less likely to be finished (per hour above 1 h)
    "priority": 0.25,  # per 2 priority steps above 3
    "category_rate": 1.5,  # user's on-time rate for this category, centred at 0.7
    "rescheduled": -0.3,  # each previous reschedule
    "late_evening": -0.3,  # sessions starting at or after 21:00 local
    "time_fit": 0.8,  # preference for this part of the day (task time-of-day model), centred at 0.25
}


@dataclass(frozen=True)
class Wellness:
    energy: float = 0.5
    focus: float = 0.5
    fatigue: float = 0.5
    burnout: float = 0.3


NEUTRAL = Wellness()


@dataclass
class CompletionFeatures:
    slack_hours: float
    energy: float
    focus: float
    fatigue: float
    duration_min: int
    priority: int
    category: TaskCategory
    category_rate: float
    times_rescheduled: int
    local_start_hour: int
    base_logit: float | None = None  # task-level logit from the trained completion model (if available)
    slack_now_hours: float | None = None  # slack if the task were done right now (reference for base_logit)
    time_pref: float | None = None  # task time-of-day model: share for this slot's part of the day


@dataclass(frozen=True)
class TaskPrior:
    """What the trained models say about a task before it is placed."""

    p_on_time: float | None = None  # completion model (XGBoost, OULAD)
    parts: dict[str, float] | None = None  # time-of-day preference (MS-LaTTE model)


def _slot_terms(f: CompletionFeatures) -> float:
    """How a particular time slot changes the chance: wellness, length, hour and time-of-day fit."""
    z = (
        COEF["energy"] * ENERGY_DEMAND.get(f.category, 1.0) * (f.energy - 0.5)
        + COEF["focus"] * FOCUS_DEMAND.get(f.category, 0.5) * (f.focus - 0.5)
        + COEF["fatigue"] * (f.fatigue - 0.5)
        + COEF["duration_h"] * (f.duration_min / 60 - 1)
        + COEF["late_evening"] * (1 if f.local_start_hour >= 21 and FOCUS_DEMAND.get(f.category, 0.5) >= 0.5 else 0)
    )
    if f.time_pref is not None:
        z += COEF["time_fit"] * (f.time_pref - 0.25)
    return z


def completion_probability(f: CompletionFeatures) -> float:
    """Chance the session is completed on time.

    With a trained-model prior (base_logit): the model's task-level logit, adjusted for this slot (wellness, length,
    hour, time-of-day fit) and for the time lost before the slot. Without it: the transparent logistic baseline
    (COEF)."""
    if f.base_logit is not None:
        ref = f.slack_now_hours if f.slack_now_hours is not None else f.slack_hours
        z = (
            f.base_logit
            + COEF["log_slack_h"] * (math.log1p(max(f.slack_hours, 0.0)) - math.log1p(max(ref, 0.0)))
            + _slot_terms(f)
        )
        if f.slack_hours < 0:
            z -= 4.0
        return 1 / (1 + math.exp(-z))
    z = (
        COEF["intercept"]
        + COEF["log_slack_h"] * math.log1p(max(f.slack_hours, 0.0))
        + COEF["energy"] * ENERGY_DEMAND.get(f.category, 1.0) * (f.energy - 0.5)
        + COEF["focus"] * FOCUS_DEMAND.get(f.category, 0.5) * (f.focus - 0.5)
        + COEF["fatigue"] * (f.fatigue - 0.5)
        + COEF["duration_h"] * (f.duration_min / 60 - 1)
        + COEF["priority"] * (f.priority - 3) / 2
        + COEF["category_rate"] * (f.category_rate - 0.7)
        + COEF["rescheduled"] * f.times_rescheduled
        + COEF["late_evening"] * (1 if f.local_start_hour >= 21 and FOCUS_DEMAND.get(f.category, 0.5) >= 0.5 else 0)
    )
    if f.time_pref is not None:
        z += COEF["time_fit"] * (f.time_pref - 0.25)
    if f.slack_hours < 0:
        z -= 4.0
    return 1 / (1 + math.exp(-z))


def category_rate(history: UserHistory, category: TaskCategory) -> float:
    if category in history.on_time_rate_by_category:
        return history.on_time_rate_by_category[category]
    return history.on_time_rate_30d if history.on_time_rate_30d is not None else 0.7


# ------------------------------------------------------------------------------------------------ grid
def ceil_to_cell(t: datetime) -> datetime:
    t = t.replace(second=0, microsecond=0) + (timedelta(minutes=1) if t.second or t.microsecond else timedelta(0))
    rem = t.minute % CELL_MIN
    return t + timedelta(minutes=(CELL_MIN - rem) % CELL_MIN)


@dataclass
class Grid:
    start: datetime
    n: int
    tz: ZoneInfo
    available: list[bool]
    wellness: list[Wellness]
    scheduled: list[bool] = field(default_factory=list)
    meals: list[tuple[datetime, datetime]] = field(default_factory=list)

    def time(self, i: int) -> datetime:
        return self.start + i * CELL

    def index_floor(self, t: datetime) -> int:
        return math.floor((t - self.start) / CELL)

    def index_ceil(self, t: datetime) -> int:
        return math.ceil((t - self.start) / CELL)

    def block(self, i: int, k: int) -> tuple[datetime, datetime]:
        return self.time(i), self.time(i + k)


def wellness_lookup(forecast: WellnessForecast | None):
    if forecast is None or not forecast.slots:
        return lambda t: NEUTRAL
    slots = sorted(forecast.slots, key=lambda s: s.start)
    starts = [s.start for s in slots]

    def at(t: datetime) -> Wellness:
        j = bisect.bisect_right(starts, t) - 1
        if j >= 0 and slots[j].start <= t < slots[j].end:
            s = slots[j]
            return Wellness(s.energy, s.focus, s.mental_fatigue, s.burnout_risk)
        return NEUTRAL

    return at


def build_grid(req: ScheduleRequest) -> Grid:
    tz = ZoneInfo(req.timezone)
    start = ceil_to_cell(max(req.horizon_start, req.as_of))
    n = max(0, math.floor((req.horizon_end - start) / CELL))
    look = wellness_lookup(req.wellness)
    available, wellness = [], []
    for i in range(n):
        t = start + i * CELL
        m = t.astimezone(tz).hour * 60 + t.astimezone(tz).minute
        available.append(req.day_start_hour * 60 <= m and m + CELL_MIN <= req.day_end_hour * 60)
        wellness.append(look(t + CELL / 2))
    g = Grid(start=start, n=n, tz=tz, available=available, wellness=wellness, scheduled=[False] * n)
    blocks = [(b.start, b.end) for b in req.busy]
    blocks += [(t.fixed_start, t.fixed_end) for t in req.tasks if t.fixed_start and t.fixed_end]
    for bs, be in blocks:
        for i in range(max(0, g.index_floor(bs)), min(n, g.index_ceil(be))):
            available[i] = False
    # Fixed study/work (a lecture, a meeting) counts as focused work, so the scheduler avoids piling more
    # focused work straight after a long one (the MAX_CONTINUOUS_MIN penalty in best_block).
    for t in req.tasks:
        if t.fixed_start and t.fixed_end and FOCUS_DEMAND.get(t.category, 0.5) >= 0.5:
            for i in range(max(0, g.index_floor(t.fixed_start)), min(n, g.index_ceil(t.fixed_end))):
                g.scheduled[i] = True
    reserve_meals(g)
    return g


def reserve_meals(g: Grid) -> None:
    """Keep a free meal break each day (Module 4 places meals there). If a lecture or meeting covers the usual
    time, the break moves to the first free 30 minutes within the next 2 hours (e.g. lunch 12:30 -> 13:00)."""
    if g.n == 0:
        return
    first = g.time(0).astimezone(g.tz).date()
    last = g.time(g.n - 1).astimezone(g.tz).date()
    day = first
    while day <= last:
        for h, mm, length in MEAL_BREAKS:
            usual = datetime(day.year, day.month, day.day, h, mm, tzinfo=g.tz)
            k = length // CELL_MIN
            for shift in range(0, MEAL_SHIFT_MAX_MIN + 1, CELL_MIN):
                i = g.index_floor(usual + timedelta(minutes=shift))
                if i + k <= 0 or i >= g.n:
                    break  # this meal time is outside the planning window
                cells = range(max(0, i), min(g.n, i + k))
                if i >= 0 and i + k <= g.n and all(g.available[c] for c in cells):
                    for c in cells:
                        g.available[c] = False
                        g.scheduled[c] = False  # a break resets back-to-back work
                    g.meals.append(g.block(i, k))
                    break
        day += timedelta(days=1)


# ------------------------------------------------------------------------------------------------ helpers
def ceil15(minutes: float) -> int:
    return int(math.ceil(minutes / CELL_MIN) * CELL_MIN)


def split_sessions(estimated_minutes: int) -> list[int]:
    if estimated_minutes <= MAX_SESSION_MIN:
        return [ceil15(estimated_minutes)]
    n = math.ceil(estimated_minutes / MAX_SESSION_MIN)
    return [ceil15(estimated_minutes / n)] * n


def urgency(task: SchedulableTask, as_of: datetime, horizon_end: datetime) -> float:
    if task.deadline is None:
        hours = (horizon_end - as_of).total_seconds() / 3600 + 48
    else:
        hours = max(0.0, (task.deadline - as_of).total_seconds() / 3600)
    return 0.6 / (1 + hours / 24) + 0.4 * task.priority / 5


def order_tasks(tasks: list[SchedulableTask], as_of: datetime, horizon_end: datetime) -> list[SchedulableTask]:
    return sorted(tasks, key=lambda t: (-urgency(t, as_of, horizon_end), -t.estimated_minutes, str(t.task_id)))


def _part_of_hour(hour: int) -> str:
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 21:
        return "evening"
    return "night"


def features_for(
    task: SchedulableTask, g: Grid, i: int, k: int, history: UserHistory, prior: TaskPrior | None = None
) -> CompletionFeatures:
    ws = g.wellness[i : i + k] or [NEUTRAL]
    start, end = g.block(i, k)
    slack = NO_DEADLINE_SLACK_H if task.deadline is None else (task.deadline - end).total_seconds() / 3600
    base_logit = slack_now = time_pref = None
    if prior is not None and prior.p_on_time is not None:
        p = min(max(prior.p_on_time, 1e-4), 1 - 1e-4)
        base_logit = math.log(p / (1 - p))
        slack_now = NO_DEADLINE_SLACK_H if task.deadline is None else (task.deadline - g.start).total_seconds() / 3600
    if prior is not None and prior.parts:
        time_pref = prior.parts.get(_part_of_hour(start.astimezone(g.tz).hour))
    return CompletionFeatures(
        slack_hours=slack,
        energy=sum(w.energy for w in ws) / len(ws),
        focus=sum(w.focus for w in ws) / len(ws),
        fatigue=sum(w.fatigue for w in ws) / len(ws),
        duration_min=k * CELL_MIN,
        priority=task.priority,
        category=task.category,
        category_rate=category_rate(history, task.category),
        times_rescheduled=task.times_rescheduled,
        local_start_hour=start.astimezone(g.tz).hour,
        base_logit=base_logit,
        slack_now_hours=slack_now,
        time_pref=time_pref,
    )


def continuous_run_min(g: Grid, i: int, k: int) -> int:
    """Length of back-to-back scheduled flexible work if cells i..i+k-1 were added."""
    left = i
    while left > 0 and g.scheduled[left - 1]:
        left -= 1
    right = i + k
    while right < g.n and g.scheduled[right]:
        right += 1
    return (right - left) * CELL_MIN


def best_block(
    task: SchedulableTask,
    g: Grid,
    k: int,
    first: int,
    history: UserHistory,
    prior: TaskPrior | None = None,
    earliest: bool = False,
) -> tuple[int, float] | None:
    last_end = g.n if task.deadline is None else min(g.n, g.index_floor(task.deadline))
    best: tuple[float, int, float] | None = None
    run_free = 0
    for i in range(first, last_end):
        run_free = run_free + 1 if g.available[i] else 0
        if run_free < k:
            continue
        s = i - k + 1
        if s < first:
            continue
        p = completion_probability(features_for(task, g, s, k, history, prior))
        if earliest:  # fallback mode: the first block that fits
            return s, p
        score = p - (0.1 if continuous_run_min(g, s, k) > MAX_CONTINUOUS_MIN else 0.0)
        if best is None or score > best[0] + 1e-9:  # strict: ties keep the earliest start
            best = (score, s, p)
    return None if best is None else (best[1], best[2])


def overload_check(req: ScheduleRequest, g: Grid, flexible: list[SchedulableTask]) -> bool:
    """Earliest-deadline-first feasibility: demand before each deadline vs free minutes before it."""
    due = sorted(
        (t for t in flexible if t.deadline is not None and req.as_of < t.deadline <= req.horizon_end),
        key=lambda t: t.deadline,
    )
    demand = 0
    for t in due:
        demand += t.estimated_minutes
        free = sum(g.available[: max(0, min(g.n, g.index_floor(t.deadline)))]) * CELL_MIN
        if demand > free:
            return True
    if req.wellness and req.wellness.slots:
        inside = [s.burnout_risk for s in req.wellness.slots if s.end > req.horizon_start and s.start < req.horizon_end]
        if inside and sum(inside) / len(inside) > BURNOUT_OVERLOAD:
            return True
    return False


# ------------------------------------------------------------------------------------------------ main
def schedule(
    req: ScheduleRequest, priors: dict[UUID, TaskPrior] | None = None, model: ModelInfo = MODEL
) -> ScheduleResult:
    """Place the tasks. `priors` (optional) carries the trained models' view of each task."""
    priors = priors or {}
    g = build_grid(req)
    items: list[ScheduledItem] = []
    unscheduled: list[UnscheduledTask] = []

    fixed = [t for t in req.tasks if t.fixed_start and t.fixed_end]
    flexible = [t for t in req.tasks if not (t.fixed_start and t.fixed_end)]
    look = wellness_lookup(req.wellness)
    for t in fixed:
        if t.fixed_end <= req.horizon_start or t.fixed_start >= req.horizon_end:
            continue
        w = look(t.fixed_start + (t.fixed_end - t.fixed_start) / 2)
        slack = 24.0 if t.deadline is None else (t.deadline - t.fixed_end).total_seconds() / 3600
        p = completion_probability(
            CompletionFeatures(
                slack,
                w.energy,
                w.focus,
                w.fatigue,
                int((t.fixed_end - t.fixed_start).total_seconds() // 60),
                t.priority,
                t.category,
                category_rate(req.history, t.category),
                t.times_rescheduled,
                t.fixed_start.astimezone(g.tz).hour,
            )
        )
        items.append(
            ScheduledItem(
                task_id=t.task_id,
                start=t.fixed_start,
                end=t.fixed_end,
                completion_probability=round(p, 4),
                is_fixed=True,
            )
        )

    overload = overload_check(req, g, flexible)

    for t in order_tasks(flexible, req.as_of, req.horizon_end):
        if t.deadline is not None and t.deadline <= req.as_of:
            unscheduled.append(UnscheduledTask(task_id=t.task_id, reason="deadline_passed"))
            continue
        placed: list[tuple[int, int, float]] = []
        # 1st try: each session in its best slot. If a later session no longer fits (an early session took a late
        # slot), roll back and retry with every session in the earliest slot that fits.
        for earliest in (False, True):
            placed = []
            first = 0
            for minutes in split_sessions(t.estimated_minutes):
                k = minutes // CELL_MIN
                found = best_block(t, g, k, first, req.history, priors.get(t.task_id), earliest=earliest)
                if found is None:
                    break
                s, p = found
                placed.append((s, k, p))
                for c in range(s, s + k):
                    g.available[c] = False
                    g.scheduled[c] = True
                first = s + k + 1  # at least a 15-min break between sessions
            if len(placed) == len(split_sessions(t.estimated_minutes)):
                break
            for s, k, _ in placed:  # roll back partial placement
                for c in range(s, s + k):
                    g.available[c] = True
                    g.scheduled[c] = False
        if len(placed) < len(split_sessions(t.estimated_minutes)):
            reason = (
                "no_free_slot_before_deadline"
                if t.deadline is not None and t.deadline <= req.horizon_end
                else "no_free_slot_in_horizon"
            )
            unscheduled.append(UnscheduledTask(task_id=t.task_id, reason=reason))
            continue
        for s, k, p in placed:
            start, end = g.block(s, k)
            items.append(ScheduledItem(task_id=t.task_id, start=start, end=end, completion_probability=round(p, 4)))

    items.sort(key=lambda x: (x.start, str(x.task_id)))
    at_risk: list[UUID] = []
    for it in sorted(items, key=lambda x: x.completion_probability):
        if not it.is_fixed and it.completion_probability < RISK_THRESHOLD and it.task_id not in at_risk:
            at_risk.append(it.task_id)
    return ScheduleResult(
        user_id=req.user_id,
        generated_at=utc_now(),
        horizon_start=req.horizon_start,
        horizon_end=req.horizon_end,
        items=items,
        unscheduled=unscheduled,
        at_risk_task_ids=at_risk,
        overload_predicted=overload or any(u.reason == "no_free_slot_before_deadline" for u in unscheduled),
        reason=req.reason,
        model=model,
    )
