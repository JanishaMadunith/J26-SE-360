"""Unit tests for the greedy scheduler (src/m3/scheduling/heuristic.py)."""

from datetime import timedelta
from itertools import pairwise
from zoneinfo import ZoneInfo

from m3.scheduling.heuristic import CompletionFeatures, completion_probability, schedule, split_sessions
from wellness_contracts import ScheduleRequest, TaskCategory

COLOMBO = ZoneInfo("Asia/Colombo")


def run(req: dict):
    return schedule(ScheduleRequest.model_validate(req))


def no_overlaps(items):
    items = sorted(items, key=lambda x: x.start)
    return all(a.end <= b.start for a, b in pairwise(items))


def test_no_tasks_gives_empty_schedule(helpers):
    res = run(helpers.request([]))
    assert res.items == [] and res.unscheduled == [] and not res.overload_predicted


def test_deadlines_respected_and_no_overlaps(helpers):
    ts = [
        helpers.task(f"t{i}", minutes=60 + 15 * i, priority=1 + i % 5, deadline=helpers.local(i % 2, 18 + i % 3))
        for i in range(6)
    ]
    req = ScheduleRequest.model_validate(helpers.request(ts))
    res = schedule(req)
    deadlines = {t.task_id: t.deadline for t in req.tasks}
    assert res.items, "something should be scheduled"
    assert all(it.end <= deadlines[it.task_id] for it in res.items)
    assert no_overlaps(res.items)


def test_items_end_before_their_deadline(helpers):
    ts = [helpers.task(f"t{i}", minutes=90, deadline=helpers.local(1, 12)) for i in range(4)]
    res = run(helpers.request(ts))
    req = ScheduleRequest.model_validate(helpers.request(ts))
    dl = {t.task_id: t.deadline for t in req.tasks}
    assert len(res.items) == 4
    assert all(it.end <= dl[it.task_id] for it in res.items)


def test_day_window_respected(helpers):
    ts = [helpers.task(f"t{i}", minutes=120) for i in range(6)]
    res = run(helpers.request(ts, day_start_hour=8, day_end_hour=20))
    for it in res.items:
        s, e = it.start.astimezone(COLOMBO), it.end.astimezone(COLOMBO)
        assert s.hour >= 8 and (e.hour < 20 or (e.hour == 20 and e.minute == 0))
        assert s.date() == (e - timedelta(minutes=1)).date()


def test_flexible_tasks_leave_meal_breaks_free(helpers):
    from m3.scheduling.heuristic import MEAL_BREAKS

    ts = [helpers.task(f"t{i}", minutes=120) for i in range(10)]  # enough work to fill whole days
    res = run(helpers.request(ts))
    assert res.items
    for it in res.items:
        s, e = it.start.astimezone(COLOMBO), it.end.astimezone(COLOMBO)
        for h, m, length in MEAL_BREAKS:
            b0 = s.replace(hour=h, minute=m)
            b1 = b0 + timedelta(minutes=length)
            assert e <= b0 or s >= b1, f"{it.start}-{it.end} overlaps meal break {h:02d}:{m:02d}"


def test_busy_blocks_respected(helpers):
    busy = [(helpers.local(0, 9), helpers.local(0, 13)), (helpers.local(0, 14), helpers.local(0, 18))]
    ts = [helpers.task("a", minutes=60, deadline=helpers.local(0, 21)), helpers.task("b", minutes=45)]
    res = run(helpers.request(ts, busy=busy))
    req = ScheduleRequest.model_validate(helpers.request(ts, busy=busy))
    for it in res.items:
        for b in req.busy:
            assert it.end <= b.start or it.start >= b.end
    assert no_overlaps(res.items)


def test_fixed_tasks_kept_exactly(helpers):
    fixed = (helpers.local(0, 10), helpers.local(0, 12))
    ts = [helpers.task("lecture", fixed=fixed), helpers.task("study", minutes=120, deadline=helpers.local(0, 13))]
    res = run(helpers.request(ts))
    req = ScheduleRequest.model_validate(helpers.request(ts))
    lec = next(i for i in res.items if i.is_fixed)
    assert lec.start == req.tasks[0].fixed_start and lec.end == req.tasks[0].fixed_end
    assert no_overlaps(res.items)


def test_high_priority_study_goes_to_high_focus_slot(helpers):
    def profile(h):  # sharp peak 15:00–17:00, otherwise tired
        return (0.9, 0.95, 0.2, 0.2) if 15 <= h < 17 else (0.35, 0.3, 0.7, 0.3)

    # one-day horizon: both tasks compete for the single 15:00–17:00 peak
    ts = [
        helpers.task("thesis writing", category="study", minutes=120, priority=5, deadline=helpers.local(0, 20)),
        helpers.task("laundry", category="chores", minutes=60, priority=2, deadline=helpers.local(0, 20)),
    ]
    res = run(helpers.request(ts, wellness=lambda uid: helpers.forecast(uid, profile), days=1))
    study = next(i for i in res.items if str(i.task_id) == ts[0]["task_id"])
    assert study.start.astimezone(COLOMBO).hour == 15
    chores = next(i for i in res.items if str(i.task_id) == ts[1]["task_id"])
    assert not (15 <= chores.start.astimezone(COLOMBO).hour < 17)


def test_long_task_split_into_sessions(helpers):
    t = helpers.task("project", minutes=300, deadline=helpers.local(1, 21))
    res = run(helpers.request([t]))
    parts = [i for i in res.items if str(i.task_id) == t["task_id"]]
    assert len(parts) == 3 and all((p.end - p.start) <= timedelta(minutes=120) for p in parts)
    assert sum((p.end - p.start).total_seconds() for p in parts) >= 300 * 60
    parts.sort(key=lambda p: p.start)
    assert all(b.start - a.end >= timedelta(minutes=15) for a, b in pairwise(parts))


def test_overload_flag_when_demand_exceeds_time(helpers):
    # 10 x 2 h due tonight, but only ~14 h available today (08:00–22:00)
    ts = [helpers.task(f"t{i}", minutes=120, deadline=helpers.local(0, 22)) for i in range(10)]
    res = run(helpers.request(ts))
    assert res.overload_predicted
    assert any(u.reason == "no_free_slot_before_deadline" for u in res.unscheduled)
    assert no_overlaps(res.items)


def test_overload_flag_from_burnout_forecast(helpers):
    res = run(
        helpers.request(
            [helpers.task("x")], wellness=lambda uid: helpers.forecast(uid, lambda h: (0.4, 0.4, 0.7, 0.85))
        )
    )
    assert res.overload_predicted


def test_deadline_passed_reported(helpers):
    t = helpers.task("late", deadline=helpers.AS_OF - timedelta(hours=1))
    res = run(helpers.request([t]))
    assert res.items == [] and res.unscheduled[0].reason == "deadline_passed"


def test_at_risk_flags_low_probability(helpers):
    bad = lambda h: (0.1, 0.1, 0.95, 0.4)  # noqa: E731
    t = helpers.task("exam prep", minutes=120, priority=1, deadline=helpers.local(0, 11), rescheduled=4)
    res = run(helpers.request([t], wellness=lambda uid: helpers.forecast(uid, bad)))
    assert res.items and res.at_risk_task_ids == [res.items[0].task_id]


def test_completion_model_is_monotonic():
    base = {
        "slack_hours": 24,
        "energy": 0.5,
        "focus": 0.5,
        "fatigue": 0.5,
        "duration_min": 60,
        "priority": 3,
        "category": TaskCategory.study,
        "category_rate": 0.7,
        "times_rescheduled": 0,
        "local_start_hour": 10,
    }
    p = completion_probability(CompletionFeatures(**base))
    assert completion_probability(CompletionFeatures(**{**base, "slack_hours": 72})) > p
    assert completion_probability(CompletionFeatures(**{**base, "focus": 0.9})) > p
    assert completion_probability(CompletionFeatures(**{**base, "fatigue": 0.9})) < p
    assert completion_probability(CompletionFeatures(**{**base, "duration_min": 180})) < p
    assert completion_probability(CompletionFeatures(**{**base, "times_rescheduled": 3})) < p
    assert completion_probability(CompletionFeatures(**{**base, "slack_hours": -1})) < 0.2


def test_split_sessions():
    assert split_sessions(50) == [60]
    assert split_sessions(120) == [120]
    assert split_sessions(300) == [105, 105, 105]


def test_deterministic(helpers):
    ts = [
        helpers.task(
            f"t{i}", minutes=45 * (1 + i % 3), priority=1 + i % 5, tid=f"00000000-0000-0000-0000-00000000000{i}"
        )
        for i in range(8)
    ]
    req = helpers.request(ts)
    a, b = run(req), run(req)
    assert [(i.task_id, i.start) for i in a.items] == [(i.task_id, i.start) for i in b.items]
