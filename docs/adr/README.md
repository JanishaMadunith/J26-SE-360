# Architecture Decision Records

Short records of decisions that are expensive to change. Add one when the team makes such a decision; never edit an accepted ADR's decision — write a new ADR that supersedes it.

| # | Decision | Status |
|---|---|---|
| [0001](0001-mobile-flutter-kotlin.md) | Mobile app in Flutter, Android-first, Kotlin for native sensing | Accepted — 17 Oct 2026 |
| [0002](0002-core-api-plus-module-services.md) | Core API plus one service per research module | Accepted — 17 Oct 2026 |
| [0003](0003-postgresql-and-own-auth.md) | PostgreSQL (TimescaleDB image) + Redis, with our own JWT auth | Accepted — 17 Oct 2026 |
| [0004](0004-server-side-processing.md) | All feature extraction and models run on the server | Accepted — 17 Oct 2026 |
| [0005](0005-monorepo-and-git-workflow.md) | One monorepo, trunk-based flow, CODEOWNERS per module | Accepted — 17 Oct 2026 |
| [0006](0006-contracts-single-source.md) | Pydantic contracts as the single source of truth | Accepted — 17 Oct 2026 |

Template: copy any ADR and keep the headings — Context, Decision, Alternatives considered, Consequences.
