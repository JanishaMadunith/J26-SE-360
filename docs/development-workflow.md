# Development workflow

How each member works day to day without needing everyone else's code running.

## The daily loop

```bash
git checkout main && git pull
git checkout -b feat/m2-sleep-features      # one branch per piece of work
make up                                     # whole system in Docker (in the background)
```

Then work on **your** part with hot reload, as below. Run `make test` before pushing, open a PR, and keep it small.

## Working on a module service (M1–M4)

Run only your service locally with auto-reload, while the rest runs in Docker:

```bash
cd services/m3-schedule-optimizer
python3.12 -m venv .venv && source .venv/bin/activate     # once
pip install -e ../../contracts -r requirements.txt pytest # once, and after requirements change
uvicorn app.main:app --reload --port 8103
```

- Open `http://localhost:8103/docs` and send a request. Each endpoint shows an example body built from the contract.
- Stop your Docker copy first so the port is free: `docker compose stop m3-schedule-optimizer`.
- To make the core API use your local copy, set `M3_URL=http://host.docker.internal:8103` in `.env` and run `docker compose up -d core-api`. Change it back afterwards.
- `make simulate` then sends a phone upload through the **whole** pipeline, including your local service.

Tests:

```bash
python -m pytest -q          # from your service folder
ruff check . && ruff format .
```

## Working on the core API

```bash
cd backend/core-api
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ../../contracts -r requirements.txt pytest
docker compose up -d postgres redis m1-affective-state m2-wellness-prediction m3-schedule-optimizer m4-food-recommendation
export DATABASE_URL=postgresql+psycopg://wellness:wellness_dev_pw@localhost:5432/wellness
alembic upgrade head
uvicorn app.main:app --reload --port 8000
```

When you change a database model:

```bash
alembic revision --autogenerate -m "add task tags"
# read the generated file in alembic/versions/ — autogenerate is not always right
alembic upgrade head
```

Commit the migration in the same PR as the model change. Teammates run `make migrate` after pulling.

## Working on the mobile app

```bash
cd mobile/wellness_app
flutter pub get
flutter run                     # phone connected; `adb reverse tcp:8000 tcp:8000` to reach the laptop backend
flutter analyze && flutter test # before pushing
```

Feature folders under `lib/features/` map to modules (see CODEOWNERS): `sensing` (M1), `insights` (M2), `tasks` and `schedule` (M3), `meals` (M4). When a contract changes, update the matching Dart model in the same PR.

## From research to production: promoting a model

Every module moves through the same three stages. Only the last one is visible to the rest of the system.

```
notebooks/             explore, plot, try ideas (not imported by anything)
    ↓  when an approach works
src/<pkg>/             data loading, features, training, evaluation as functions + tests
    ↓  when the model beats the baseline on a held-out set
app/predictor.py       loads models/<artifact> and serves it behind Predictor.predict()
```

1. **Notebook.** Quick experiments. Keep notebooks small and numbered (`01-eda.ipynb`, `02-baseline.ipynb`). Clear outputs that contain data before committing.
2. **`src/<pkg>/`.** Move working code into functions: `features.py`, `train.py`, `evaluate.py`. Add unit tests for feature maths. Training scripts read datasets from `DATA_DIR` and write artifacts to `models/`.
3. **Evaluate honestly.** Hold out data the model never sees during training or tuning (e.g. a later cohort or later time period). Report the held-out numbers in `models/model_card.json`.
4. **Serve.** In `app/predictor.py`, load the artifact at start-up and keep the baseline as a fallback if the file is missing. The request and response contracts don't change, so nobody else is affected.
5. **PR.** Include the model card and a short results note. Share the artifact as a Release asset (see CONTRIBUTING §8).

## Before you push — checklist

- [ ] `make test` and `make lint` pass
- [ ] If you touched `contracts/`: `make contracts` run, consumers updated, all owners tagged
- [ ] If you added a DB model change: migration included
- [ ] No data, `.env`, models or notebook outputs with data committed
- [ ] PR title is a Conventional Commit, e.g. `feat(m2): add sleep-duration feature`
