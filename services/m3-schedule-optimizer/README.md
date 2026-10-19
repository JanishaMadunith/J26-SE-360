# M3 — AI-Based Predictive Adaptive Schedule Optimization

**Owner:** G.W.D.D.N. Kumarasinghe (IT23267336) · **Port:** 8103 · **Endpoint:** `POST /v1/schedule` · `GET /health` ·
**Demo:** `GET /demo`

Request `ScheduleRequest` → response `ScheduleResult`, defined once in [`contracts/`](../../contracts). The core API is the
only caller. The service is **stateless**: no database, no user data stored.

## What it does (v0.2.0)

| Part | Data | Code | Result |
|---|---|---|---|
| **Completion model** — chance a task is finished on time | OULAD (663,420 cases) | `src/m3/train_oulad.py`, served by `src/m3/completion_model.py` | XGBoost: **83.4 %** accuracy, ROC-AUC **0.916** on the unseen 2014J semester; time CV 84.0 % ± 1.4 |
| **Time-of-day model** — when people usually do a task | MS-LaTTE (10,101 titles) | `src/m3/tasktime.py` | 53.7 % (4 classes; most-common guess 41.6 %), top-2 82.8 % |
| **Wellness stand-in** for Module 2 | PMData (1,724 days, 16 people) | `src/m3/context_datasets.py` | hourly energy / focus / fatigue / burnout in Module 2's format |
| **Default busy hours** | ATUS 2025 weekday diaries | `src/m3/context_datasets.py` | work / class hours for a working adult and a university student |
| **Wellness evidence** | StudentLife (44 students) | `src/m3/context_datasets.py` | +0.18 stress and −0.25 h sleep per extra deadline (within student) |
| **Scheduler** | — | `src/m3/scheduling/heuristic.py` | places each task in the slot with the best adjusted chance |

How a schedule is made (`app/predictor.py`):
1. The **completion model** gives each flexible task a chance of being finished on time from the user's behaviour
   (on-time rate, recent activity, unfinished / overdue tasks; from the task-event log) and the task (time left,
   priority).
2. The **time-of-day model** gives each task title a preference over morning / afternoon / evening / night.
3. The **scheduler** splits the horizon into 15-minute cells (day window, busy blocks, fixed tasks and meal breaks
   removed), orders tasks by urgency, and puts each session where the model's chance — adjusted for that slot's
   wellness (Module 2 forecast), length, hour, time-of-day fit and the time lost before it — is highest.
4. Flags: `at_risk_task_ids` (chance < 0.5), `unscheduled` with a reason, `overload_predicted`.

If the trained files are missing (fresh clone, CI), the service falls back to the **transparent baseline** (hand-set
logistic model, `COEF` in `heuristic.py`) and `/health` says so. The baseline is also the comparison baseline for the
evaluation. **Next:** OR-Tools CP-SAT optimiser behind the same interface (`src/m3/scheduling/cpsat.py`), simulation on
ATUS / PMData / StudentLife users, then pilot data.

## Build the models (needs the datasets in `../../../07-datasets`)

```bash
cd services/m3-schedule-optimizer
source .venv/bin/activate                     # python -m pip install -r requirements-train.txt
PYTHONPATH=src python -m m3.train_oulad       # completion model (long: tuning) -> models/, reports/
PYTHONPATH=src python -m m3.build_demo_assets # defaults + real held-out cases for the demo
PYTHONPATH=src python -m m3.tasktime          # time-of-day model (MS-LaTTE)
PYTHONPATH=src python -m m3.context_datasets  # PMData, ATUS, StudentLife
# evidence in MLflow:
PYTHONPATH=src python -m m3.log_to_mlflow && PYTHONPATH=src python -m m3.log_model_history && PYTHONPATH=src python -m m3.cross_validate
mlflow ui --backend-store-uri sqlite:///mlflow.db      # http://127.0.0.1:5000
```

## Run, demo and test

```bash
PYTHONPATH=src:. uvicorn app.main:app --port 8103    # http://localhost:8103/demo  and  /docs
python -m pytest                                     # model tests skip themselves if models/ is empty
```

## Folder layout

```
m3-schedule-optimizer/
├── app/          serving: main.py (FastAPI), predictor.py, demo.py (model + demo endpoints), static/demo.html
├── src/m3/       research + training code (imported by the app — same code for training and serving)
├── notebooks/    evidence notebook (01_completion_model_evidence.ipynb)
├── models/       trained artefacts — gitignored except model_card.json and the small demo JSON files
├── reports/      results JSON + figures
├── tests/        pytest (contract, behaviour and model tests)
└── Dockerfile    built from the repo root (needs ../../contracts)
```

Data never goes in the repo: read it from `DATA_DIR` (the `07-datasets` folder next to the repo).

## Rules

* Keep the `Predictor` class name, its `info: ModelInfo` and `predict(request) -> result` signature — the core API depends on them.
* Never change a request/response shape here; change it in `contracts/`, run `make contracts`, and tell the team.
* Every model change bumps `info.version` — it is stored with every prediction.
