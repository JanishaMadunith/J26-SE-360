## What and why

<!-- One or two sentences. Link the issue: "Closes #12" -->

## Scope

- [ ] core
- [ ] m1
- [ ] m2
- [ ] m3
- [ ] m4
- [ ] mobile
- [ ] contracts
- [ ] docs / ci

## Checklist

- [ ] PR title follows Conventional Commits, e.g. `feat(m3): add OR-Tools optimizer`
- [ ] Small and focused (ideally < ~400 changed lines)
- [ ] Tests added or updated, and `make test` passes locally
- [ ] Lint passes (`make lint`, and `flutter analyze` for mobile)
- [ ] Docs / README updated if behaviour or setup changed

## Contract impact

- [ ] No contract change
- [ ] Contract changed **compatibly** (new optional fields only) — `make contracts` run, consumers updated
- [ ] **Breaking** contract change — all owners tagged, version bumped, `!` in title

## Privacy check

- [ ] No raw content collected (no typed characters, no video frames, no plain app names)
- [ ] No participant data, datasets, `.env` or secrets in this PR
- [ ] Any new collected signal is documented in `docs/data-and-privacy.md`

## Mobile only

<!-- Screenshots or a short screen recording of the changed screens -->

## Model changes only

- [ ] `models/model_card.json` updated (data, features, metrics, threshold, date)
- [ ] Artifact shared via Release asset / Git LFS (not committed)
- [ ] Baseline fallback still works if the artifact is missing
