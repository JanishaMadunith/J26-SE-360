# API and contracts

Every payload below is a Pydantic model in [`contracts/src/wellness_contracts/`](../contracts/src/wellness_contracts/). JSON Schemas for each one are generated into [`contracts/schemas/`](../contracts/schemas/) — the mobile app uses them as the reference for its Dart models.

Live, clickable docs are always at `http://localhost:<port>/docs` once `make up` is running. This page is the overview.

## Conventions

- JSON over HTTPS. Times are **ISO 8601 with timezone** (`2026-10-20T09:00:00+05:30`); naive times are rejected.
- IDs are UUIDs.
- Unknown fields are **rejected** (`extra="forbid"`) — a typo fails loudly instead of being ignored.
- Scores (`focus`, `energy`, `completion_probability`, …) are floats from 0 to 1.
- Errors use FastAPI's format: `{"detail": "..."}` with 400 / 401 / 404 / 409 / 422.
- Core API endpoints (except auth and health) need `Authorization: Bearer <access_token>`.

## Core API — port 8000, prefix `/v1`

### Auth and profile

| Method | Path | Body | Response |
|---|---|---|---|
| POST | `/v1/auth/register` | `{email, password (≥ 8 chars), display_name, timezone}` | 201 `{access_token, refresh_token, token_type: "bearer", user}` |
| POST | `/v1/auth/login` | `{email, password}` (JSON) | same token response |
| POST | `/v1/auth/refresh` | `{refresh_token}` | same token response |
| GET | `/v1/me` | — | `User {id, email, display_name, timezone, created_at, consent_version, consented_at}` |
| POST | `/v1/me/consent` | `{version}` | `User` |
| GET / PUT | `/v1/me/diet-profile` | `DietProfile` | `DietProfile` |

### Sensing

| Method | Path | Body | Response |
|---|---|---|---|
| POST | `/v1/sensing/batches` | [`SensorBatch`](../contracts/schemas/sensor_batch.schema.json) | 202 [`SensorBatchAck`](../contracts/schemas/sensor_batch_ack.schema.json) — `duplicate: true` if the `batch_id` was already received |
| POST | `/v1/self-reports` | `SelfReport` | 201 |

### Tasks

| Method | Path | Body | Response |
|---|---|---|---|
| GET | `/v1/tasks?status=pending` | — | `[Task]` |
| POST | `/v1/tasks` | [`TaskCreate`](../contracts/schemas/task_create.schema.json) | 201 [`Task`](../contracts/schemas/task.schema.json) |
| PATCH | `/v1/tasks/{id}` | [`TaskUpdate`](../contracts/schemas/task_update.schema.json) | `Task` |
| POST | `/v1/tasks/{id}/complete` | — | `Task` |
| DELETE | `/v1/tasks/{id}` | — | 204 (task becomes `cancelled`) |

Every task change is written to a task-event log (`created`, `scheduled`, `completed`, `missed`, `rescheduled`, …). That log is Module 3's future training data.

### Schedule, insights, recommendations

| Method | Path | Response |
|---|---|---|
| GET | `/v1/schedule/current` | [`ScheduleResult`](../contracts/schemas/m3_schedule_result.schema.json) (404 if none yet) |
| POST | `/v1/schedule/regenerate` | `ScheduleResult` — runs **M3 only**, using the cached wellness forecast |
| GET | `/v1/insights/latest` | `{"affective": AffectiveStateResult \| null, "wellness": WellnessForecast \| null}` |
| GET | `/v1/recommendations/latest` | [`Recommendations`](../contracts/schemas/m4_recommendations.schema.json) (404 if none yet) |

### Pipeline and health

| Method | Path | Response |
|---|---|---|
| POST | `/v1/pipeline/run` | `PipelineRun {id, status: "ok" \| "partial" \| "failed", steps: [StepStatus], started_at, finished_at}` |
| GET | `/v1/pipeline/runs?limit=20` | `[PipelineRun]` |
| GET | `/health` | `{"status": "ok" \| "degraded", "db": bool, "modules": {"m1": bool, "m2": bool, "m3": bool, "m4": bool}}` |

## Module services

All four expose `GET /health → {"status": "ok", "service": "<name>", "model": {"name", "version"}}` and one prediction endpoint. They are called **only by the core API**, never by the app.

| Module | Port | Endpoint | Request | Response |
|---|---|---|---|---|
| M1 Affective state | 8101 | `POST /v1/affective-state` | [`AffectiveStateRequest`](../contracts/schemas/m1_affective_state_request.schema.json) — raw signals for one window | [`AffectiveStateResult`](../contracts/schemas/m1_affective_state_result.schema.json) — label, probabilities, confidence, extracted features |
| M2 Wellness | 8102 | `POST /v1/wellness-forecast` | [`WellnessForecastRequest`](../contracts/schemas/m2_wellness_forecast_request.schema.json) — affective history, 48 h sleep/activity signals, workload | [`WellnessForecast`](../contracts/schemas/m2_wellness_forecast.schema.json) — slots of `slot_minutes` with fatigue, focus, energy, burnout risk |
| M3 Schedule | 8103 | `POST /v1/schedule` | [`ScheduleRequest`](../contracts/schemas/m3_schedule_request.schema.json) — tasks, busy blocks, wellness, user history | [`ScheduleResult`](../contracts/schemas/m3_schedule_result.schema.json) — items with completion probability, unscheduled, at-risk, overload flag |
| M4 Food | 8104 | `POST /v1/recommendations` | [`RecommendationRequest`](../contracts/schemas/m4_recommendation_request.schema.json) — state, wellness, schedule, diet profile | [`Recommendations`](../contracts/schemas/m4_recommendations.schema.json) — meals + suggested activities |

Each result carries `model: {name, version}`, so every stored prediction can be traced to the model that produced it.

## Versioning

- The contracts package has a version in `contracts/src/wellness_contracts/__init__.py` (currently `0.1.0`).
- **Compatible changes** (new optional fields, new enum values that consumers can ignore) are a minor bump and need only the affected owners' review.
- **Breaking changes** (rename, remove, change a type or meaning) are a major bump, need all four owners' approval, and must update every consumer in the same PR. See [CONTRIBUTING.md §5](../CONTRIBUTING.md#5-changing-a-contract-safely).
- The URL prefix `/v1` changes only if we ever need to run two incompatible API versions side by side. For this project, we don't expect to.
