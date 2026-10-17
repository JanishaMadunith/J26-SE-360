"""Paths and constants for Module 3.

DATA_DIR points at the shared `07-datasets` folder that sits NEXT TO the repo (data never goes in git):

    Reserch/
    ├── 07-datasets/                      <- DATA_DIR
    └── J26-SE-360/                       <- repo root
        └── services/
            └── m3-schedule-optimizer/    <- MODULE_ROOT
                └── src/m3/config.py

Override with the DATA_DIR environment variable (e.g. DATA_DIR=/path/to/07-datasets).
"""

import os
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[2]  # services/m3-schedule-optimizer/
DATA_DIR = Path(os.environ.get("DATA_DIR", MODULE_ROOT.parents[2] / "07-datasets"))  # sibling of the repo
MODELS_DIR = Path(os.environ.get("M3_MODELS_DIR", MODULE_ROOT / "models"))
REPORTS_DIR = Path(os.environ.get("M3_REPORTS_DIR", MODULE_ROOT / "reports"))

SEED = 42

# OULAD temporal split: train on earlier cohorts, test on the latest one.
TRAIN_PRESENTATIONS = ["2013B", "2013J", "2014B"]
VALID_PRESENTATION = "2014B"  # used only for tuning (train on 2013B + 2013J)
TEST_PRESENTATION = "2014J"  # never seen during training or tuning

# Prediction horizons: days before the deadline at which the prediction is made.
HORIZONS = (1, 3, 7, 14)
