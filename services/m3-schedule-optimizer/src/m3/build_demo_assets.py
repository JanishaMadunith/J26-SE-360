"""Build the small files the M3 service needs to serve the completion model and the demo.

  models/completion_defaults.json  training-set medians (numbers) and categories, used to fill inputs the app
                                   does not provide
  models/demo_examples.json        real held-out 2014J cases (never used in training) with their true outcome,
                                   so the demo can show prediction vs. what really happened

Usage (from services/m3-schedule-optimizer):
    PYTHONPATH=src python -m m3.build_demo_assets [--samples-cache path/to/oulad_samples.pkl]
"""

import argparse
import json

import numpy as np
import pandas as pd

from .completion_model import CompletionModel
from .config import MODELS_DIR, SEED, TEST_PRESENTATION, TRAIN_PRESENTATIONS
from .features_oulad import CATEGORICAL, FEATURE_SETS

F = FEATURE_SETS["behaviour"]
SUMMARY = [
    "prior_same_type_on_time_rate",
    "prior_on_time_rate",
    "days_since_last",
    "active_7",
    "clicks_7",
    "n_prior_outstanding",
    "n_overdue_all",
]


def _clean(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return round(float(v), 4)
    return v if isinstance(v, (int, float)) else str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-cache", default=None)
    ap.add_argument("--n", type=int, default=24)
    a = ap.parse_args()
    if a.samples_cache:
        df = pd.read_pickle(a.samples_cache)
    else:
        from .data_oulad import load_oulad
        from .features_oulad import build_dataset

        print("Building the OULAD samples from 07-datasets (a few minutes)...")
        df = build_dataset(load_oulad())

    tr = df[df.code_presentation.isin(TRAIN_PRESENTATIONS)]
    cats = [c for c in F if c in CATEGORICAL]
    defaults = {
        "median": {c: _clean(tr[c].median()) for c in F if c not in cats},
        "categories": {c: [str(x) for x in tr[c].cat.categories] for c in cats},
        "mode": {c: str(tr[c].mode().iloc[0]) for c in cats},
        "source": f"median of the training cohorts {', '.join(TRAIN_PRESENTATIONS)} ({len(tr):,} samples)",
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "completion_defaults.json").write_text(json.dumps(defaults, indent=1))
    print(f"wrote {MODELS_DIR / 'completion_defaults.json'}")

    # real held-out cases at 3 and 7 days before the deadline
    te = df[(df.code_presentation == TEST_PRESENTATION) & df.H.isin([3, 7])].copy()
    model = CompletionModel()
    sample = te.sample(min(len(te), 20000), random_state=SEED)
    rows = [{c: _clean(r[c]) for c in F} for _, r in sample[F].iterrows()]
    sample["p"] = model.predict(rows)
    thr = model.threshold
    sample["pred"] = (sample.p >= thr).astype(int)
    q = a.n // 4
    picks = pd.concat(
        [
            sample[(sample.y == 1) & (sample.p > 0.85)].head(q),  # clearly on time, correct
            sample[(sample.y == 0) & (sample.p < 0.15)].head(q),  # clearly at risk, correct
            sample[(sample.p > 0.4) & (sample.p < 0.65)].head(q),  # borderline
            sample[sample.pred != sample.y].head(q),  # honest mistakes
        ]
    ).drop_duplicates(subset=["id_student", "id_assessment", "H"])
    out = []
    for i, (_, r) in enumerate(picks.sample(frac=1, random_state=SEED).iterrows(), 1):
        out.append(
            {
                "id": f"case-{i:02d}",
                "course": str(r.code_module),
                "semester": str(r.code_presentation),
                "task": "Tutor-marked assignment (big)" if r.is_tma == 1 else "Computer-marked quiz (small)",
                "weight": _clean(r.weight),
                "days_before_deadline": int(r.H),
                "actual_on_time": bool(r.y),
                "summary": {c: _clean(r[c]) for c in SUMMARY},
                "features": {c: _clean(r[c]) for c in F},
            }
        )
    (MODELS_DIR / "demo_examples.json").write_text(json.dumps(out, indent=1))
    acc = float((sample.pred == sample.y).mean())
    print(f"wrote {len(out)} demo cases; model accuracy on this {len(sample):,}-case test sample: {acc:.3f}")


if __name__ == "__main__":
    main()
