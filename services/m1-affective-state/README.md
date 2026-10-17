# M1 — Multimodal Affective State Detection

**Owner:** J.D. Jayatilake (IT23279698) · **Port:** 8101 · **Endpoint:** `POST /v1/affective-state` · `GET /health`

Request `AffectiveStateRequest` → response `AffectiveStateResult` — defined once in [`contracts/`](../../contracts) (JSON Schemas:
`contracts/schemas/m1_affective_state_*.schema.json`). The core API is the only caller. The service is **stateless**: no database,
no user data stored.

## What the baseline does (v0.1.0)

* **Server-side feature extraction** (`src/m1_affect/features.py`) from the raw signals the phone uploads:
  * **PPG** → resample to the camera frame rate, detrend, 0.7–3.5 Hz Butterworth band-pass on the red or green channel
    (whichever has the stronger pulse band), heart rate from the Welch spectrum (refined with detected beats) and
    **HRV RMSSD** from inter-beat intervals. Recordings under 10 s are ignored.
  * **Typing** → keys/min, mean/std inter-key interval (pauses > 5 s excluded), backspace rate, mean hold time.
  * **Phone use** → unlocks/hour, screen-on minutes, app switches/hour, share of social/video/game time.
  * **Activity / light** → active minutes, steps/hour, mean ambient light.
* **Rule-based classifier** (`app/predictor.py`): each feature is standardised against resting reference values, five
  transparent rule scores (stress, anxiety, relaxed, happy, sad) are combined with a softmax. Missing signals contribute
  nothing, so the service always answers. All extracted features are returned in `features` for logging.

The baseline is deliberately simple and transparent so the whole app works end-to-end from day one. It is also the
**comparison baseline** for the trained model in your evaluation chapter.

## What you replace

1. Collect labelled data (pilot app self-reports are the labels) and/or prepare public data (K-EmoPhone, SWELL-KW).
2. Build feature tables with `m1_affect.features.extract_all()` so training and serving compute features the same way.
3. Train and evaluate with **subject-wise** cross-validation (never mix one person's data across train/test).
4. Save the model to `models/` with a `model_card.json`; load it in `Predictor.__init__` and keep `predict()`'s signature.
5. **Open issue:** agree the final class set (5 classes in the Topic Assessment Form vs 3 elsewhere) and change it only
   in `contracts/` (`AffectiveLabel`).

## Folder layout

```
m1-affective-state/
├── app/               serving only: main.py (FastAPI), predictor.py (Predictor.predict), settings.py
├── src/m1_affect/   research + training code (imported by the app — same code for training and serving)
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
cd services/m1-affective-state
../../.venv/bin/python -m pytest                     # tests
../../.venv/bin/uvicorn app.main:app --reload --port 8101   # http://localhost:8101/docs
```

Or everything together: `make up` (Docker Compose) from the repo root. `make test-m1` runs only these tests.

## Rules

* Keep the `Predictor` class name, its `info: ModelInfo` and `predict(request) -> result` signature — the core API depends on them.
* Never change a request/response shape here; change it in `contracts/`, run `make contracts`, and tell the team.
* Every model change bumps `info.version` — it is stored with every prediction.
