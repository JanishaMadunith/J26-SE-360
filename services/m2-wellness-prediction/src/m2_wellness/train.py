"""Train M2 models on StudentLife multimodal dataset and report metrics."""
import argparse
import json
import pathlib

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
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


def evaluate_model(model, train_features, test_features, train, test, train_key, test_key):
    results = {}
    for target in TARGETS:
        model.fit(train_features, train[target])
        pred = np.clip(model.predict(test_features), 0, 1)
        base = test_key.map(train.groupby(train_key)[target].mean()).fillna(train[target].mean())
        errors = np.abs(test[target].to_numpy() - pred)
        results[target] = {
            "mae": round(float(mean_absolute_error(test[target], pred)), 4),
            "r2": round(float(r2_score(test[target], pred)), 4),
            "within_10_points_percentage": round(float(np.mean(errors <= 0.10) * 100), 2),
            "baseline_mae": round(float(mean_absolute_error(test[target], base)), 4),
            "baseline_r2": round(float(r2_score(test[target], base)), 4),
        }
    return results


def main(compare_random_forest=False):
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

    results = evaluate_model(
        HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=0),
        train_features,
        test_features,
        train,
        test,
        train_key,
        test_key,
    )
    for target in TARGETS:
        print("hist_gradient_boosting", target, results[target])
        model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=0)
        model.fit(train_features, train[target])
        joblib.dump(model, MODELS / f"{target}.joblib")

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

    if compare_random_forest:
        random_forest_results = evaluate_model(
            RandomForestRegressor(
                n_estimators=300,
                max_features="sqrt",
                min_samples_leaf=2,
                random_state=0,
                n_jobs=-1,
            ),
            train_features,
            test_features,
            train,
            test,
            train_key,
            test_key,
        )
        comparison = {
            "split": "same user-level GroupShuffleSplit as HistGradientBoosting",
            "hist_gradient_boosting": results,
            "random_forest": random_forest_results,
        }
        (MODELS / "model_comparison.json").write_text(json.dumps(comparison, indent=2))
        for target in TARGETS:
            print("random_forest", target, random_forest_results[target])
        print("Saved comparison to", MODELS / "model_comparison.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compare-random-forest",
        action="store_true",
        help="Train Random Forest separately without replacing production artifacts.",
    )
    main(compare_random_forest=parser.parse_args().compare_random_forest)
