# ADR-0006: Pydantic contracts are the single source of truth

**Status:** Accepted — 17 Oct 2026

## Context

Seven components exchange data: the app, the core API and four module services, plus training code that must produce the same shapes. Earlier team documents already disagreed on details: three vs five affective classes, daily vs hourly wellness forecasts, and which direction data flows between M3 and M4. Interfaces described only in documents drift quickly.

## Decision

- Every message is defined **once**, as a Pydantic v2 model in `contracts/src/wellness_contracts/` (package `wellness-contracts`).
- The core API and all module services **import** these models. They don't redefine them.
- Models reject unknown fields (`extra="forbid"`) and require timezone-aware datetimes, so mistakes fail loudly.
- JSON Schemas are **generated** into `contracts/schemas/` with `make contracts`. CI and a pre-commit hook fail if they are out of date. The Flutter app uses these schemas as the reference for its Dart models.
- Contract changes follow [CONTRIBUTING.md §5](../../CONTRIBUTING.md#5-changing-a-contract-safely): prefer additive changes; breaking changes need all owners and update every consumer in the same PR.

## Alternatives considered

| Option | Why not |
|---|---|
| OpenAPI YAML written by hand | Second copy of the truth next to the Python code. Easy to forget to update. |
| Protocol Buffers / gRPC | Strong typing, but extra tooling and a steeper learning curve. HTTP/JSON is easier to debug and demo. |
| Each service defines its own models | Fastest at first, but the disagreements above would end up in code. |

## Consequences

- One edit updates validation everywhere in Python; generated schemas keep the app aligned.
- The contract package is a shared dependency. Every service installs it (`pip install -e ../../contracts`, or `/contracts` in Docker).
- Dart models are written by hand from the schemas, so mobile must be updated in the same PR as a contract change. CODEOWNERS makes all owners review those PRs.
