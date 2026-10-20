"""PP1 demo scenarios and edge cases, run through the same endpoints as the /demo page.

Scenario 1 (normal day), 2 (deadline crunch), 3 (re-planning), plus: no deadline, passed deadline, same title twice,
empty list, more work than free time. Checks that need the trained models skip themselves when models/ is empty."""

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.predictor import Predictor

PRED = Predictor()
client = TestClient(create_app(PRED))
needs_models = pytest.mark.skipif(
    not (PRED.has_completion and PRED.has_tasktime), reason="trained models not in models/"
)


def T(title, category, minutes, priority, due_day):  # noqa: N802
    return {
        "title": title,
        "category": category,
        "estimated_minutes": minutes,
        "priority": priority,
        "due_day": due_day,
    }


S1 = [
    T("Morning run", "exercise", 45, 3, 0),
    T("Team meeting", "work", 60, 4, 0),
    T("Buy groceries", "chores", 45, 3, 1),
    T("Review lecture notes", "study", 90, 3, 2),
    T("Watch a movie", "personal", 120, 1, None),
]
S2 = [
    T("Assignment 1 report", "study", 240, 5, 1),
    T("Project work", "study", 180, 5, 1),
    T("Quiz preparation", "study", 120, 4, 2),
    T("Submit lab sheet", "study", 60, 5, 0),
    T("Gym", "exercise", 60, 2, 0),
]
BASE = {"persona": "university_student", "wellness_day": "typical", "days": 3}


def plan(tasks, **kw):
    r = client.post("/v1/demo/plan", json={**BASE, **kw, "tasks": tasks})
    assert r.status_code == 200, r.text
    return r.json()


def by_title(p):
    return {t["title"]: t for t in p["tasks"]}


def sessions(p):
    return [{"task_key": t["task_key"], **s} for t in p["tasks"] for s in t["sessions"]]


@needs_models
def test_scenario1_normal_day():
    p = plan(S1)
    t = by_title(p)
    assert t["Morning run"]["best_part"] == "morning"
    assert t["Team meeting"]["best_part"] == "afternoon"
    assert t["Buy groceries"]["best_part"] == "evening"
    assert t["Watch a movie"]["best_part"] == "night"
    assert p["summary"]["at_risk"] == 0 and p["summary"]["unscheduled"] == 0 and not p["overload_predicted"]
    assert t["Morning run"]["sessions"][0]["start"][11:13] < "12"  # placed in the morning


@needs_models
def test_scenario2_deadline_crunch_flags_risk():
    p1, p2 = plan(S1), plan(S2)
    assert p2["summary"]["planned_hours"] == 11.0
    assert p2["summary"]["at_risk"] >= 1  # some tasks get a low on-time chance
    assert "StudentLife" in p2["wellness"]["source"]  # deadline load lowers wellness
    mean = lambda p: sum(s["p"] for s in sessions(p)) / len(sessions(p))  # noqa: E731
    assert mean(p2) < mean(p1)


def test_scenario3_missed_task_replans_with_minimal_moves():
    p = plan(S1)
    prev = sessions(p)
    missed = next(t for t in p["tasks"] if t["title"] == "Review lecture notes")
    r = client.post(
        "/v1/demo/replan",
        json={
            "plan": {**BASE, "tasks": S1},
            "previous": prev,
            "change": {"kind": "missed", "task_key": missed["task_key"]},
        },
    )
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["replan"]["mode"] == "minimal" and j["replan"]["moved_tasks"] == []
    new = by_title(j)["Review lecture notes"]["sessions"]
    assert new and new[0]["start"] >= missed["sessions"][0]["end"]  # re-placed after the missed slot
    others = {(s["task_key"], s["start"]) for s in prev if s["task_key"] != missed["task_key"]}
    assert others <= {(s["task_key"], s["start"]) for s in sessions(j)}  # everything else kept


def test_scenario3_urgent_task_added():
    p = plan(S1)
    r = client.post(
        "/v1/demo/replan",
        json={
            "plan": {**BASE, "tasks": S1},
            "previous": sessions(p),
            "change": {"kind": "add", "new_task": T("Urgent client fix", "work", 120, 5, 0)},
        },
    )
    assert r.status_code == 200, r.text
    j = r.json()
    urgent = by_title(j)["Urgent client fix"]
    assert urgent["sessions"] or urgent["unscheduled"]
    assert j["replan"]["mode"] in ("minimal", "full")


def test_edge_cases():
    p = plan(
        [
            T("Read a novel", "personal", 60, 2, None),
            T("Pay last month's bill", "chores", 30, 4, -1),
            T("Lab report", "study", 90, 4, 1),
            T("Lab report", "study", 90, 3, 3),
        ]
    )
    t = p["tasks"]
    assert t[0]["deadline"] is None and t[0]["sessions"]  # no deadline: still planned
    assert t[1]["unscheduled"] == "deadline_passed" and not t[1]["sessions"]
    assert t[2]["task_key"] != t[3]["task_key"] and t[2]["sessions"] and t[3]["sessions"]  # same title, both kept


def test_empty_list():
    p = plan([])
    assert p["tasks"] == [] and p["summary"]["tasks"] == 0 and not p["overload_predicted"]


def test_overload_is_reported_not_crashed():
    p = plan([T(f"Big assignment {i}", "study", 480, 4, 1) for i in range(5)], days=2)
    assert p["overload_predicted"]
    assert p["summary"]["unscheduled"] >= 1
    assert all(t["unscheduled"] in (None, "no_free_slot_before_deadline") for t in p["tasks"])


def test_plan_shows_module3_contract_json():
    from wellness_contracts import ScheduleRequest, ScheduleResult

    ct = plan(S1)["contract"]
    assert ct["endpoint"] == "POST /v1/schedule"
    ScheduleRequest.model_validate(ct["request"])  # JSON in: valid Module 3 request
    ScheduleResult.model_validate(ct["response"])  # JSON out: valid Module 3 response


def test_lecture_8_to_1_is_fixed_and_lunch_moves():
    p = plan(
        [
            T("Lecture", "study", 300, 5, 0) | {"fixed": "08:00-13:00"},
            T("Assignment draft", "study", 180, 5, 1),
            T("Gym", "exercise", 60, 2, 0),
        ],
        persona="free",
    )
    lec = by_title(p)["Lecture"]["sessions"]
    assert len(lec) == 1 and lec[0]["start"][11:16] == "08:00" and lec[0]["end"][11:16] == "13:00"
    for t in p["tasks"]:
        if t["title"] != "Lecture":
            for s in t["sessions"]:
                assert s["end"] <= lec[0]["start"] or s["start"] >= lec[0]["end"]  # nothing during the lecture
    day1 = lec[0]["start"][:10]
    lunch = [m for m in p["meals"] if m["start"][:10] == day1 and "12:00" <= m["start"][11:16] <= "15:00"]
    assert lunch and lunch[0]["start"][11:16] == "13:00"  # lunch moved from 12:30 to right after the lecture
    assert not by_title(p)["Lecture"]["at_risk"]  # a fixed event is attended, not rated as at risk


def test_click_a_session_marks_it_missed():
    p = plan(S1)
    prev = sessions(p)
    run = by_title(p)["Watch a movie"]
    clicked = run["sessions"][-1]
    r = client.post(
        "/v1/demo/replan",
        json={
            "plan": {**BASE, "tasks": S1},
            "previous": prev,
            "change": {"kind": "missed", "task_key": run["task_key"], "session_start": clicked["start"]},
        },
    )
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["replan"]["missed_slot"]["start"][:16] == clicked["start"][:16]
    new = by_title(j)["Watch a movie"]
    assert new["unscheduled"] or new["sessions"][0]["start"] >= clicked["end"]
