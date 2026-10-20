# Module 3 demo — scenarios and edge cases

Start: `PYTHONPATH=src:. uvicorn app.main:app --port 8113` → http://localhost:8113/demo (tab **2 · Plan my days**).
"Today" is the next weekday from 07:00; tasks are due at 23:00 on their day. The same checks run automatically in
`tests/test_scenarios.py`.

| Scenario | How to run it on the page | What should happen | Why |
|---|---|---|---|
| **1 · Normal day** | Click **1 · Normal day** | Run → morning, meeting → afternoon, groceries → evening; "best time of day" for the movie is night. **0 at risk.** | Time-of-day model (MS-LaTTE) + light workload. The movie lands in the early evening: the scheduler weighs its night preference against free time and energy. |
| **2 · Deadline crunch** | Click **2 · Deadline crunch** | 11 h of work in ~2 days: chances drop, **some sessions marked at risk** (late-evening ones). Working adult: also **overload** and a task **not placed**. | Wellness is lowered by the deadline load using the StudentLife effects (+0.18 stress, −0.25 h sleep per deadline in the next 3 days). |
| **3 · Re-planning** | After scenario 1, pick **Review lecture notes → Mark as missed**, or **Add urgent task for today** | Banner: **Mode: minimal**, all other sessions **kept**, moved: nothing. The missed slot is outlined in red; the task is re-placed after it. | `POST /v1/demo/replan`: other sessions are locked; only the changed task is placed. If it cannot fit, the rest of the plan is rebuilt (mode "full") and moved tasks are listed. |
| **Edge cases** | Click **Edge cases** | No-deadline task planned; passed deadline → "not placed: deadline passed"; two "Lab report" tasks with different deadlines both planned. | |
| **Empty list** | Click **Empty list** | "No tasks — nothing to plan", no error. | |
| **Overload** | Click **Overload** | ⚠ overload banner; tasks that cannot fit are "not placed: no free slot before deadline" — no crash. | |

## Tab 1 · Will it be done on time?

* Filter real unseen OULAD cases by **14 / 7 / 3 / 1** day(s) before the deadline; compare the prediction with what
  really happened, with the SHAP reasons.
* **What if** sliders: past on-time rate, days since active, active days, unfinished / overdue tasks, **usually
  finishes early/late**, days left, priority — move between low and high and watch the chance and reasons change.

## Assumption to state

An app task's "days left" is converted to OULAD's horizon by how small the task is compared with an assignment
(~8 h): a 45-min errand due tonight is not as risky as an unsubmitted assignment due tonight
(`completion_model.effective_horizon`). It will be replaced by fine-tuning on pilot task logs.
