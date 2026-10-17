# Core API

FastAPI service that the mobile app talks to. It owns accounts, tasks and raw sensor data, and orchestrates
the four module services:

```
phone ──raw signals──▶ core API ──▶ M1 affective state ──▶ M2 wellness forecast ──▶ M3 schedule ──▶ M4 recommendations
                          │                                                                     │
                          └──────────── PostgreSQL (every input and every module output) ◀──────┘
```

Every payload exchanged with the app or a module is defined once in [`/contracts`](../../contracts)
(`wellness_contracts`). This service never redefines them.

**Owner:** tech lead (Dilsara). Changes go through a PR like everything else.

## Endpoints

All `/v1` routes except `/v1/auth/*` need `Authorization: Bearer <access_token>`. Interactive docs: `http://localhost:8000/docs`.

| Method | Path | Body → Response | Notes |
|---|---|---|---|
| POST | `/v1/auth/register` | `{email, password, display_name, timezone}` → tokens + user | 201; 409 if email exists |
| POST | `/v1/auth/login` | `{email, password}` → tokens + user | 401 on bad credentials |
| POST | `/v1/auth/refresh` | `{refresh_token}` → tokens + user | access token = 60 min, refresh = 30 days |
| GET | `/v1/me` | → user | |
| POST | `/v1/me/consent` | `{version}` → user | records research consent |
| GET / PUT | `/v1/me/diet-profile` | `DietProfile` | used by Module 4 |
| POST | `/v1/sensing/batches` | `SensorBatch` → `SensorBatchAck` | 202; idempotent on `batch_id`; may start the pipeline in the background |
| POST | `/v1/self-reports` | `SelfReport` → `SelfReport` | 201; mood/stress/energy/focus check-in |
| GET | `/v1/tasks?status=` | → `[Task]` | earliest deadline first |
| POST | `/v1/tasks` | `TaskCreate` → `Task` | 201 |
| PATCH | `/v1/tasks/{id}` | `TaskUpdate` → `Task` | status changes are logged |
| POST | `/v1/tasks/{id}/complete` | → `Task` | logs `completed` with `on_time` |
| DELETE | `/v1/tasks/{id}` | → 204 | soft delete (`cancelled`) — history is kept for research |
| GET | `/v1/schedule/current` | → `ScheduleResult` | 404 until the first schedule |
| POST | `/v1/schedule/regenerate` | → `ScheduleResult` | runs **M3 only** with the latest stored forecast; 503 if M3 is down |
| GET | `/v1/insights/latest` | → `{affective, wellness}` | either may be `null` |
| GET | `/v1/recommendations/latest` | → `Recommendations` | 404 until the first run |
| POST | `/v1/pipeline/run` | → `PipelineRun` | runs M1→M4 now (dev/demo) |
| GET | `/v1/pipeline/runs?limit=` | → `[PipelineRun]` | newest first |
| GET | `/health` | → `{status, db, modules}` | `ok` or `degraded` |

## How the pipeline works

`app/services/pipeline.py` runs, for one user:

1. **Overdue tasks → `missed`** (logged in `task_events`).
2. **M1** with the raw signals of the last `PIPELINE_WINDOW_MINUTES` (skipped if there is no new data).
3. **M2** with 7 days of M1 results, 48 h of screen/activity/steps/light, self-reports and the current workload.
4. **M3** with open tasks (7-day horizon), the forecast and the user's on-time history. Scheduled tasks become
   `scheduled`; moved slots are logged as `rescheduled`.
5. **M4** with the latest affective state, forecast, schedule and diet profile.

Each step is recorded as a `StepStatus`. If a module fails, the **last stored result is reused**
(`used_fallback: true`) and the run continues — run status is `ok`, `partial` or `failed`.
The run starts automatically after a sensor upload when the previous run is older than
`PIPELINE_MIN_INTERVAL_MINUTES`; `POST /v1/pipeline/run` starts it on demand.

## Data model

| Table | What it holds |
|---|---|
| `users`, `devices`, `diet_profiles` | accounts, phones, food preferences |
| `sensor_batches` | every upload, verbatim (`payload`) — idempotency key = `batch_id` |
| `sensor_events` | long-format events: `keystroke`, `screen`, `app_usage`, `activity`, `steps`, `light` (TimescaleDB hypertable when available) |
| `ppg_recordings`, `self_reports` | camera heart-rate signals; check-ins (ground-truth labels) |
| `tasks`, `task_events` | tasks + append-only event log (Module 3's future training data) |
| `affective_states`, `wellness_forecasts`, `schedules`, `recommendations` | each module output as full contract JSON (`payload`) with model name/version |
| `pipeline_runs` | every orchestration run and its step statuses |

## Run locally

With Docker (recommended) — from the repo root: `docker compose up core-api` (starts PostgreSQL too).

Without Docker:

```bash
cd backend/core-api
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ../../contracts -r requirements-dev.txt
export DATABASE_URL=postgresql+psycopg://wellness:wellness_dev_pw@localhost:5432/wellness
alembic upgrade head
uvicorn app.main:app --reload --port 8000
```

Module URLs default to `http://localhost:8101` … `8104`; set `M1_URL` … `M4_URL` to change them.

## Tests

```bash
pytest                         # in-memory SQLite, ~15 s, module services are faked
```

Against PostgreSQL (what CI should run) — use a **separate** test database; tests empty every table:

```bash
DATABASE_URL=postgresql+psycopg://wellness:wellness_dev_pw@localhost:5432/wellness_test alembic upgrade head
TEST_DATABASE_URL=postgresql+psycopg://wellness:wellness_dev_pw@localhost:5432/wellness_test pytest
```

The fake modules live in `tests/conftest.py` (`FakeModules`); add a module name to `fake_modules.down`
to simulate an outage.

## Migrations

```bash
alembic revision --autogenerate -m "describe the change"   # after editing app/models.py
alembic upgrade head
alembic check                                              # CI: fails if models and migrations differ
```

Review every autogenerated file before committing. `0002_timescale` turns `sensor_events` into a hypertable
only when the TimescaleDB extension exists, so plain PostgreSQL works too.

## Lint

```bash
ruff check . && ruff format --check .
```
