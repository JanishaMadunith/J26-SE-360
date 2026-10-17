# src/m2_wellness — research & training code (Module 2)

| File | Purpose |
|---|---|
| `heuristics.py` | **Done (baseline).** Sleep estimate, circadian + homeostatic model, workload and stress context. |
| `datasets.py` | *To add.* Build time-series windows (affective history, sleep, activity, workload) with labels from self-reports. Candidate public data: StudentLife, PMData, LifeSnaps, GLOBEM. |
| `train.py` | *To add.* Train the forecaster (e.g. LSTM / temporal CNN / gradient boosting on lag features); save to `../models/` with `model_card.json`. |
| `evaluate.py` | *To add.* Per-indicator MAE/RMSE by horizon, compared with this baseline. |

Output must stay **slot-level** (`WellnessSlot` per `slot_minutes`) because Module 3 schedules within the day.
Data never goes in the repo — read it from `DATA_DIR` (the `07-datasets` folder next to the repo).
