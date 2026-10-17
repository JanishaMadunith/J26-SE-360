# ADR-0005: One monorepo, trunk-based workflow, CODEOWNERS per module

**Status:** Accepted — 17 Oct 2026

## Context

Four people build one integrated system: a mobile app, a core API, four module services and shared contracts. A change to a contract must update every consumer at once. Each member also needs a clearly separated area of work.

## Decision

- **One repository** for everything (app, backend, services, contracts, docs). Datasets stay outside it in the sibling `07-datasets/` folder.
- **Trunk-based development:** `main` is protected, work happens on short-lived branches (`feat/m3-…`, `fix/…`, `docs/…`) merged through PRs with **one approval** and green CI.
- **CODEOWNERS:** each member owns their module folder and matching app feature folder. Everyone owns `contracts/`, `docker-compose.yml` and `.github/`.
- **Conventional Commits** with module scopes (`feat(m2): …`) for a readable history.
- **CI per path:** GitHub Actions runs only the jobs for changed parts. A contract change re-tests every consumer.
- **Weekly integration:** every Friday, merge and run `make up && make simulate` on a fresh clone.

## Alternatives considered

| Option | Why not |
|---|---|
| One repo per module | Contract changes need coordinated PRs in five repos. Integration testing is harder. Duplicated tooling. |
| Long-lived module branches merged at milestones | "Merge hell" before every presentation. Integration problems found late. |
| Pushing straight to `main` | No review, easy to break the shared pipeline right before a demo. |

## Consequences

- One clone gives the whole system; `make up` runs it.
- Branch protection and CODEOWNERS have to be enabled by the repo owner in GitHub settings (Settings → Branches → protect `main`: require PR, 1 approval, code-owner review, status checks).
- Members must keep PRs small and rebase often.
