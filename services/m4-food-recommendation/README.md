# M4 — Context-Aware Personalized Food Recommendation

**Owner:** T.C. Gunaweera (IT23174658) · **Port:** 8104 · **Endpoint:** `POST /v1/recommendations` · `GET /health`

Request `RecommendationRequest` → response `Recommendations` — defined once in [`contracts/`](../../contracts) (JSON Schemas:
`contracts/schemas/m4_recommendation*.schema.json`). The core API is the only caller. The service is **stateless**: no database,
no user data stored.

## What the baseline does (v0.1.0)

* **Meal windows**: for breakfast 07:00, lunch 12:30, snack 16:00 and dinner 19:30 (local) within the next 24 h, the
  nearest free 30-min gap in the Module 3 schedule (±90 min); otherwise the usual time with a note.
* **Filters** from the diet profile: vegetarian / vegan / halal, allergies and dislikes, diabetes → no high-GI foods,
  hypertension → no high-sodium foods.
* **Context scoring**: stress/anxiety → omega-3, magnesium, complex carbs, no caffeine/fried/sugary foods; sad → omega-3,
  complex carbs, protein; low forecast energy → protein + complex carbs; no caffeine after 15:00; calories near the
  per-meal share of the daily target; variety penalty for repeats within the day.
* **Food table**: ~30 Sri Lankan and general foods (`src/m4_food/foods.py`) with typical calories and tags.
* **Suggested activities** (max 2): short walk when stressed, rest break when fatigue is high, planning break when the
  schedule is overloaded. Shown to the user only — if accepted they become normal tasks for Module 3 (no M4 → M3 data flow).

The baseline is deliberately simple and transparent so the whole app works end-to-end from day one. It is also the
**comparison baseline** for the trained model in your evaluation chapter.

## What you replace

1. Replace the food table with a proper nutrition knowledge base (Sri Lankan food composition tables, USDA FoodData Central).
2. Learn a ranker from user feedback (accepted / skipped meals) and evaluate recommendation quality + satisfaction.
3. Keep `predict()`'s signature and the meal/activity output contract.

## Folder layout

```
m4-food-recommendation/
├── app/               serving only: main.py (FastAPI), predictor.py (Predictor.predict), settings.py
├── src/m4_food/     research + training code (imported by the app — same code for training and serving)
├── notebooks/         exploration (clear outputs before committing)
├── models/            trained artefacts — gitignored except model_card.json
├── tests/             pytest (contract + behaviour tests)
├── requirements.txt   serving dependencies
└── Dockerfile         built from the repo root (needs ../../contracts)
```

Data never goes in the repo: read it from `DATA_DIR` (the `07-datasets` folder next to the repo).

## Run and test

```bash
# from the repo root, once:  make setup
cd services/m4-food-recommendation
../../.venv/bin/python -m pytest                     # tests
../../.venv/bin/uvicorn app.main:app --reload --port 8104   # http://localhost:8104/docs
```

Or everything together: `make up` (Docker Compose) from the repo root. `make test-m4` runs only these tests.

## Rules

* Keep the `Predictor` class name, its `info: ModelInfo` and `predict(request) -> result` signature — the core API depends on them.
* Never change a request/response shape here; change it in `contracts/`, run `make contracts`, and tell the team.
* Every model change bumps `info.version` — it is stored with every prediction.
