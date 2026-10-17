# ADR-0002: Core API plus one service per research module

**Status:** Accepted — 17 Oct 2026

## Context

Four members each own one research module with different ML stacks: signal processing and classifiers (M1), time-series forecasting (M2), XGBoost plus OR-Tools optimisation (M3), recommendation (M4). Each member is assessed individually, so their module must be clearly theirs, testable alone and demonstrable alone. The modules also have to work together as one framework.

## Decision

- A **core API** (FastAPI) owns everything shared: auth, users, consent, tasks and the task-event log, raw sensor ingest, the database, and orchestration of the pipeline M1 → M2 → M3 → M4.
- Each module is a **separate, stateless FastAPI service** (`services/mX-…`) with one prediction endpoint and `/health`. It receives everything it needs in the request and returns a contract-typed result. It never touches the database.
- **Docker Compose** runs everything with one command (`make up`).
- Every service ships a simple **baseline predictor** first, so the whole pipeline works from day one. Members replace the baseline with their trained model behind the same `Predictor.predict()` interface.

## Alternatives considered

| Option | Why not |
|---|---|
| One FastAPI app with four module packages | Everyone shares one dependency set (TensorFlow vs XGBoost vs OR-Tools pins clash). One person's import error breaks everyone. Ownership is blurrier. |
| Firebase backend + Cloud Run ML services | Less backend code, but ties the project to Google's platform. Research data export and SQL analysis are harder. |
| Modules calling each other directly | Creates hidden dependencies and makes a module untestable alone. Central orchestration keeps data flow one-way and visible. |

## Consequences

- Each member can run, test and demo their module in isolation (`uvicorn app.main:app --reload`).
- Clear boundary for marking: a module's folder is its owner's work.
- More moving parts than a monolith. Mitigated by Docker Compose, a shared service template and the same contract package everywhere.
- Network calls between services add latency. Mitigated by timeouts, cached last results and an M3-only regenerate path (see [architecture.md](../architecture.md#fallback-behaviour)).
