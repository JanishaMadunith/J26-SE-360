# M2 — AI-Based Cognitive Wellness Prediction

**Owner:** J.M. Samarathunga (IT23401662) · **Port:** 8102 · **Endpoint:** `POST /v1/wellness-forecast` · `GET /health`

Request `WellnessForecastRequest` → response `WellnessForecast` — defined once in [`contracts/`](../../contracts) (JSON Schemas:
`contracts/schemas/m2_wellness_forecast*.schema.json`). The core API is the only caller. The service is **stateless**: no database,
no user data stored.

## What the baseline does (v0.1.0)

* **Sleep estimate**: longest night-time screen-off gap in the last 36 h (overlapping 21:00–11:00 local), minus time
  with recorded steps; bright rooms (mean lux > 30) count 80 %.
* **Two-process alertness model**: circadian curve (peak ≈ 16:00, post-lunch dip) + homeostatic fatigue
  `1 − exp(−hours_awake/18)` + sleep debt vs 7.5 h.
* **Context**: workload due in 24 h, overdue tasks, Module 1 stress/anxiety share (24 h and 7 days), and the latest
  self-reported energy (blended in, decaying over ~6 h).
* Outputs **mental fatigue, focus, energy, burnout risk** (0–1) for every slot (`slot_minutes`, default 60) aligned to the
  user's local clock — slot-level output is what Module 3 needs.

The baseline is deliberately simple and transparent so the whole app works end-to-end from day one. It is also the
**comparison baseline** for the trained model in your evaluation chapter.

## What you replace

1. Build time-series training windows (affective history, sleep, activity, phone use, workload) with self-report labels
   (pilot data; public candidates: StudentLife, PMData, LifeSnaps, GLOBEM).
2. Train the forecaster (e.g. LSTM / temporal CNN / gradient boosting on lag features); evaluate MAE per indicator and
   horizon **against this baseline**.
3. Save to `models/` with a `model_card.json`; load it in `Predictor.__init__`; keep the slot-level output.

## Folder layout

```
m2-wellness-prediction/
├── app/               serving only: main.py (FastAPI), predictor.py (Predictor.predict), settings.py
├── src/m2_wellness/ research + training code (imported by the app — same code for training and serving)
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
cd services/m2-wellness-prediction
../../.venv/bin/python -m pytest                     # tests
../../.venv/bin/uvicorn app.main:app --reload --port 8102   # http://localhost:8102/docs
```

Or everything together: `make up` (Docker Compose) from the repo root. `make test-m2` runs only these tests.

## Rules

* Keep the `Predictor` class name, its `info: ModelInfo` and `predict(request) -> result` signature — the core API depends on them.
* Never change a request/response shape here; change it in `contracts/`, run `make contracts`, and tell the team.
* Every model change bumps `info.version` — it is stored with every prediction.
