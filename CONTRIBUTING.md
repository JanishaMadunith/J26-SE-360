# Contributing

How the four of us work in one repository without breaking each other's work.

## 1. Branches

- `main` is **protected**: no direct pushes, merge only through a pull request with green CI.
- Create a short-lived branch for each piece of work and merge it within a few days:

| Prefix | Use for | Example |
|---|---|---|
| `feat/<scope>-…` | New functionality | `feat/m3-ortools-optimizer` |
| `fix/<scope>-…` | Bug fixes | `fix/core-token-refresh` |
| `docs/…` | Documentation only | `docs/setup-windows` |
| `chore/…` | Tooling, dependencies, CI | `chore/ci-cache-pip` |
| `research/<scope>-…` | Experiments you may not merge | `research/m2-lstm-baseline` |

Keep your branch up to date with `git pull --rebase origin main` before opening a PR.

## 2. Commit messages

We use [Conventional Commits](https://www.conventionalcommits.org) with a module scope:

```
<type>(<scope>): <short summary in present tense>
```

- **Types:** `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`, `ci`
- **Scopes:** `core`, `m1`, `m2`, `m3`, `m4`, `mobile`, `contracts`, `docs`, `ci`

Examples:

```
feat(m1): extract HRV (RMSSD) from PPG signal
fix(core): return 404 when no schedule exists yet
feat(contracts)!: rename WellnessSlot.energy to energy_level
```

A `!` after the scope marks a **breaking** change. Breaking contract changes need agreement from every module owner (see §5).

## 3. Pull requests

- **Small PRs** — ideally under ~400 changed lines. Split big work.
- Fill in the PR template checklist.
- **1 approval** required. CODEOWNERS automatically asks the owner of every folder you touched.
- **CI must be green** (lint + tests for every part you changed).
- Changes under `contracts/`, `docker-compose.yml` or `.github/` need review from **all owners whose module consumes them** — in practice, all four of us.
- Mobile UI changes: add a screenshot or short screen recording.
- Squash-merge, keeping a Conventional Commit title.

## 4. Who owns what

See [.github/CODEOWNERS](.github/CODEOWNERS). Short version: each member owns their `services/mX-…` folder and the matching feature folder in the app. The tech lead (M3) reviews `backend/core-api`. Everyone co-owns `contracts/`.

You may open PRs anywhere, but the owner has the final say on their module.

## 5. Changing a contract safely

Every message between the app, the core API and the modules is defined in `contracts/src/wellness_contracts/`. That Python package is the single source of truth. The JSON Schemas in `contracts/schemas/` are **generated** from it.

1. Open an issue describing the change and tag every affected owner.
2. Edit the Pydantic model. Prefer **adding optional fields** over renaming or removing fields — that keeps old clients working.
3. Run `make contracts` to regenerate the JSON Schemas. CI fails if you forget.
4. Update every consumer in the **same PR**: core API, the affected module services, and the mobile models.
5. Label the PR `contracts` and request review from all owners.
6. For a breaking change, bump `__version__` in `contracts/src/wellness_contracts/__init__.py` and mention it in the PR title with `!`.

## 6. Code style

| Language | Tool | Command |
|---|---|---|
| Python | ruff (lint + format), config in `ruff.toml` | `make lint` / `ruff format .` |
| Dart / Flutter | `flutter analyze`, `dart format` | `cd mobile/wellness_app && flutter analyze` |
| Kotlin | Android Studio formatter; ktlint optional | — |
| Markdown, YAML | pre-commit hooks | runs on `git commit` |

`make setup` installs pre-commit hooks, so most problems are caught before you push. Don't bypass them with `--no-verify`.

## 7. Tests expected with each change

| You change… | Add or update… |
|---|---|
| A core API endpoint | An API test in `backend/core-api/tests/` (runs on SQLite and on PostgreSQL in CI) |
| A module's predictor | A test that a sample request returns a valid contract response (`services/mX-…/tests/`) |
| Feature extraction or training code | Unit tests for the maths on small hand-made inputs |
| A contract | Validation tests for the new rules + every consumer's tests still pass |
| Mobile logic (repositories, state) | A Dart unit test; widget test for important screens |

Run `make test` before pushing. It runs every Python test suite.

## 8. Adding a trained model

1. Training code lives in `services/mX-…/src/<pkg>/`. Notebooks are for exploration only — move anything you rely on into `src/`.
2. Save artifacts to `services/mX-…/models/`. That folder is **gitignored** except `model_card.json`.
3. Commit `models/model_card.json` with: model name and version, training data, features, metrics on the held-out set, decision threshold, date, and how to reproduce it.
4. Share the artifact itself through a **GitHub Release asset** (preferred) or Git LFS. Never commit files over 1 MB — a pre-commit hook blocks them.
5. Load the artifact in `app/predictor.py` behind the same `Predictor.predict()` interface. Keep the baseline as a fallback if the file is missing.

## 9. Research data rules

- **Never commit data** — no CSVs, pickles, exports, or participant data, even "small samples".
- Datasets live in the sibling folder `../07-datasets` (outside the repo). Code reads them through the `DATA_DIR` environment variable.
- Pilot data from real participants is handled as described in [docs/data-and-privacy.md](docs/data-and-privacy.md). It never leaves the agreed storage.
- Never paste participant data into issues, PRs or chat.

## 10. Weekly integration ritual (every Friday)

1. Everyone merges finished work into `main` before the meeting.
2. One person runs from a fresh clone: `make up && make simulate`.
3. Check the pipeline output: every step `ok`, no fallbacks used unexpectedly.
4. Fix anything broken first; nothing new merges until `main` works end to end.
5. Note decisions in `docs/open-issues.md` or a new ADR in `docs/adr/`.
