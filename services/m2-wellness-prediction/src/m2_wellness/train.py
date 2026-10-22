"""Train M2 models on StudentLife multimodal dataset and report metrics."""
import json
import pathlib

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

try:
    from .datasets import load_or_build_dataset
    from .features import FEATURES, TARGETS
except ImportError:
    from m2_wellness.datasets import load_or_build_dataset
    from m2_wellness.features import FEATURES, TARGETS

MODELS = pathlib.Path(__file__).resolve().parents[2] / "models"


def key(d):
    return d.hour_sin.round(2).astype(str) + "_" + d.hour_cos.round(2).astype(str)


def main():
    MODELS.mkdir(exist_ok=True)
    df = load_or_build_dataset()

    # split by USER so no person appears in both train and test
    tr, te = next(GroupShuffleSplit(test_size=0.2, random_state=0).split(df, groups=df.user_id))
    train, test = df.iloc[tr], df.iloc[te]
    train_features = train[FEATURES].replace([np.inf, -np.inf], np.nan)
    test_features = test[FEATURES].replace([np.inf, -np.inf], np.nan)
    medians = train_features.median().fillna(0.0)
    train_features = train_features.fillna(medians)
    test_features = test_features.fillna(medians)

    # baseline: average per time-of-day, no user context
    train_key, test_key = key(train), key(test)

    results = {}
    for t in TARGETS:
        model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=0)
        model.fit(train_features, train[t])
        pred = np.clip(model.predict(test_features), 0, 1)

        base = test_key.map(train.groupby(train_key)[t].mean()).fillna(train[t].mean())
        results[t] = {
            "mae": round(float(mean_absolute_error(test[t], pred)), 4),
            "r2": round(float(r2_score(test[t], pred)), 4),
            "baseline_mae": round(float(mean_absolute_error(test[t], base)), 4),
            "baseline_r2": round(float(r2_score(test[t], base)), 4),
        }
        print(t, results[t])
        joblib.dump(model, MODELS / f"{t}.joblib")

    meta = {
        "features": FEATURES,
        "targets": TARGETS,
        "train_rows": len(train),
        "test_rows": len(test),
        "data": "synthetic StudentLife-derived multimodal dataset",
        "metrics": results,
    }
    (MODELS / "metrics.json").write_text(json.dumps(meta, indent=2))

    card = {
        "model_name": "m2-hgb-slot-forecaster",
        "version": "0.3.0",
        "data_type": "synthetic",
        "metrics": results,
    }
    (MODELS / "model_card.json").write_text(json.dumps(card, indent=2))
    print("Saved models, metrics, and model_card to", MODELS)


if __name__ == "__main__":
    main()
