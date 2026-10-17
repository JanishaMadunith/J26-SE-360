# M3 — AI-Based Predictive Adaptive Schedule Optimization

**Owner:** G.W.D.D.N. Kumarasinghe (IT23267336) · **Port:** 8103 · **Endpoint:** `POST /v1/schedule` · `GET /health`

Request `ScheduleRequest` → response `ScheduleResult` — defined once in [`contracts/`](../../contracts) (JSON Schemas:
`contracts/schemas/m3_schedule_*.schema.json`). The core API is the only caller. The service is **stateless**: no database,
no user data stored.

## What the baseline does (v0.1.0)

**Energy-aware greedy scheduler** (`src/m3/scheduling/heuristic.py`, pure functions, 15 unit tests):
1. **Grid** — horizon split into 15-min cells; a cell is free if it is inside the local day window
   (`day_start_hour`–`day_end_hour`), after `as_of`, and not overlapping a busy block or a fixed task. Each cell carries the
   Module 2 forecast (neutral profile if missing).
2. **Fixed tasks** (lectures, meetings) stay exactly where they are.
3. **Order** flexible tasks by urgency `0.6/(1 + hours_to_deadline/24) + 0.4·priority/5`; tasks > 120 min are split into
   ≤ 120-min sessions with ≥ 15 min between them.
4. **Place** each session in the contiguous free block (ending before its deadline) with the highest **completion
   probability** from a transparent logistic model: slack to deadline, slot energy and focus (weighted by how much the task
   category needs them), fatigue, duration, priority, the user's on-time rate for that category, times rescheduled, late
   evening. A soft penalty avoids more than 3 h of back-to-back work.
5. **Flags** — `at_risk_task_ids` (probability < 0.5), `unscheduled` with a reason, and `overload_predicted` when
   earliest-deadline-first demand exceeds the free time before some deadline or mean forecast burnout risk > 0.7.

The baseline is deliberately simple and transparent so the whole app works end-to-end from day one. It is also the
**comparison baseline** for the trained model in your evaluation chapter.

## What you replace

1. **Completion model**: train on the OULAD proxy task (`src/m3/train_oulad.py`, leakage-audited) and later on pilot
   task logs (the core API's task-event log); map it onto app features and swap it in for `completion_probability()`.
2. **OR-Tools CP-SAT optimiser** (`src/m3/scheduling/cpsat.py` has the design) behind the same `schedule()` interface;
   keep the heuristic as the evaluation baseline.
3. **Proactive regeneration** — the core API calls `POST /v1/schedule` with a `reason` (e.g. `task_added`,
   `task_missed`, `overload_predicted`); add stability (minimise moves vs the current plan) in the CP-SAT objective.

## Folder layout

```
m3-schedule-optimizer/
├── app/               serving only: main.py (FastAPI), predictor.py (Predictor.predict), settings.py
├── src/m3/          research + training code (imported by the app — same code for training and serving)
├── notebooks/         exploration (clear outputs before committing)
├── models/            trained artefacts — gitignored except model_card.json
├── tests/             pytest (contract + behaviour tests)
├── requirements.txt   serving dependencies
└── Dockerfile         built from the repo root (needs ../../contracts)
```

Data never goes in the repo: read it from `DATA_DIR` (the `07-datasets` folder next to the repo).

## Run and test

```bash
# from the repo root, once:  make setup
cd services/m3-schedule-optimizer
../../.venv/bin/python -m pytest                     # tests
../../.venv/bin/uvicorn app.main:app --reload --port 8103   # http://localhost:8103/docs
```

Or everything together: `make up` (Docker Compose) from the repo root. `make test-m3` runs only these tests.

## Rules

* Keep the `Predictor` class name, its `info: ModelInfo` and `predict(request) -> result` signature — the core API depends on them.
* Never change a request/response shape here; change it in `contracts/`, run `make contracts`, and tell the team.
* Every model change bumps `info.version` — it is stored with every prediction.
