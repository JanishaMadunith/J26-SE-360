# src/m3 — research & training code (Module 3)

| Path | Purpose |
|---|---|
| `scheduling/heuristic.py` | **Serving baseline.** Energy-aware greedy scheduler + transparent completion model. |
| `scheduling/cpsat.py` | **Planned.** OR-Tools CP-SAT optimiser — design notes. |
| `config.py` | Paths (`DATA_DIR` defaults to the `07-datasets` folder next to the repo) and split constants. |
| `data_oulad.py` | Loads OULAD from `DATA_DIR/oulad`. |
| `features_oulad.py` | Leakage-safe task-completion dataset: prediction at deadline − H days (H ∈ {1,3,7,14}). |
| `train_oulad.py` | Optuna-tuned LightGBM / XGBoost, temporal split (train 2013B–2014B, test 2014J), resumable tuning. |
| `report_figures.py` | Charts for the results report. |

Training needs `requirements-train.txt`:

```bash
pip install -r requirements-train.txt
DATA_DIR=/path/to/07-datasets PYTHONPATH=src python -m m3.train_oulad --lgbm-trials 30 --xgb-trials 20
DATA_DIR=/path/to/07-datasets python tests/research/test_no_leakage.py     # independent leakage audit
```
