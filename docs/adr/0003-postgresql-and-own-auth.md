# ADR-0003: PostgreSQL + Redis, with our own JWT authentication

**Status:** Accepted — 17 Oct 2026

## Context

The system stores user accounts, tasks, a high volume of time-stamped raw signals (keystroke timings, screen events, PPG recordings) and every module output. The research needs SQL queries over time windows, reproducible exports for analysis, and control over where participant data lives for the ethics application.

## Decision

- **PostgreSQL 16** as the only database, run from the **TimescaleDB** image. The time-series tables can become hypertables, but the schema must also work on plain PostgreSQL (the hypertable step is optional and guarded).
- **SQLAlchemy 2** ORM and **Alembic** migrations in the core API. Tests run on SQLite for speed and on PostgreSQL in CI.
- **Redis** for caching the latest module outputs and fallback data. Redis holds nothing that can't be rebuilt from PostgreSQL.
- **Own authentication** in the core API: bcrypt password hashing, short-lived JWT access tokens (60 min) plus refresh tokens (30 days), secret from `JWT_SECRET`.

## Alternatives considered

| Option | Why not |
|---|---|
| Supabase (hosted Postgres + auth) | Fast to start, but participant data would sit on a third-party host, which complicates ethics approval. Less control over the auth flow. |
| Firebase Auth + Firestore | Document store is weak for time-window queries and analysis. Exports are awkward. |
| MongoDB | Flexible, but the data is relational (users, tasks, events) and the analysis is SQL-shaped. |

## Consequences

- Full control over research data. One `pg_dump` or SQL export gives everything for analysis.
- We maintain auth ourselves. It's kept minimal and standard (bcrypt + JWT) and covered by tests.
- Deployment later (e.g. Azure) needs a managed PostgreSQL or a VM. For the project and viva, Docker Compose on a laptop is enough.
