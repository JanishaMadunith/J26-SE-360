"""Put EVERY model built on the way to the final model into MLflow, each as a real, loadable model.

`m3.log_to_mlflow` records the final run with its results and tuning history. This script adds the models
themselves: every baseline, the earlier feature version, the tuned candidates, the rejected ensemble and the
feature-set ablations. Each one is trained or loaded the same way `m3.train_oulad` does and evaluated on the
same held-out 2014J test cohort. Each is logged with:

  * the model file (MLflow model format, so it can be reloaded and run),
  * its settings and training data, plus a note on why it was built and what it showed,
  * test metrics, a confusion matrix and a ROC curve,
  * a numbered version in the Model Registry under "m3-task-completion". The final model is the last version
    and carries the alias "champion".

The tuned LightGBM and XGBoost models are loaded from the saved files in models/ rather than retrained, so they
are byte-for-byte the models in the report.

Usage (from services/m3-schedule-optimizer, after m3.log_to_mlflow):
    PYTHONPATH=src python -m m3.log_model_history                  # builds samples from 07-datasets
    PYTHONPATH=src python -m m3.log_model_history --samples-cache ../../../07-datasets/oulad_samples.pkl
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
# ruff: noqa: N803, N806  -- ML convention: X (features), F (feature list) are uppercase

import argparse
import json
import time
import warnings

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.metrics import confusion_matrix, roc_curve  # noqa: E402

from .config import MODELS_DIR, MODULE_ROOT, REPORTS_DIR, TEST_PRESENTATION, TRAIN_PRESENTATIONS  # noqa: E402
from .features_oulad import EXTRA_V2, FEATURE_SETS  # noqa: E402
from .train_oulad import REFIT_FACTOR, lgbm_fit, logreg, metrics, rule_baseline  # noqa: E402

EXPERIMENT = "M3 - Task completion prediction (OULAD)"
REGISTRY = "m3-task-completion"
F = FEATURE_SETS["behaviour"]
F_V1 = [c for c in F if c not in EXTRA_V2]
warnings.filterwarnings("ignore", category=UserWarning)


# ----------------------------------------------------------------------------- pyfunc wrappers
class RuleModel(mlflow.pyfunc.PythonModel):
    """Baseline rule: on time if mostly on time before; with no history, top half of cohort for recent clicks."""

    def predict(self, context, model_input, params=None):
        return rule_baseline(model_input)


class AverageModel(mlflow.pyfunc.PythonModel):
    """Average of the tuned LightGBM and XGBoost probabilities (candidate rejected on validation)."""

    def __init__(self, code_modules):
        self.code_modules = list(code_modules)

    def load_context(self, context):
        self.lgbm = lgb.Booster(model_file=context.artifacts["lightgbm"])
        self.xgb = xgb.XGBClassifier()
        self.xgb.load_model(context.artifacts["xgboost"])

    def predict(self, context, model_input, params=None):
        model_input = model_input.copy()
        model_input["code_module"] = pd.Categorical(model_input["code_module"].astype(str), self.code_modules)
        return (self.lgbm.predict(model_input) + self.xgb.predict_proba(model_input)[:, 1]) / 2


# ----------------------------------------------------------------------------- figures
def _figures(y, p, thr, title):
    cm = confusion_matrix(y, (p >= thr).astype(int), labels=[0, 1])
    f1, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}\n{cm[i, j] / cm.sum():.1%}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    ax.set_xticks([0, 1], ["not on time", "on time"])
    ax.set_yticks([0, 1], ["not on time", "on time"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(f"{title}\nconfusion matrix (test 2014J, threshold {thr:.3f})", fontsize=9)
    f1.tight_layout()
    f2, ax = plt.subplots(figsize=(4.6, 4))
    if len(np.unique(p)) > 1:
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, lw=2)
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(f"{title}\nROC curve (test 2014J)", fontsize=9)
    f2.tight_layout()
    return f1, f2


# ----------------------------------------------------------------------------- one model -> one run
def _log(step, slug, title, why, flavor, model, X_test, y_test, p_test, params, thr=0.5, extra_tags=None, **kw):
    with mlflow.start_run(run_name=f"{step:02d}-{slug}", nested=True) as run:
        mlflow.set_tags({"kind": "model version", "step": step, "mlflow.note.content": why, **(extra_tags or {})})
        mlflow.log_params({"model": title, "decision_threshold": round(thr, 3), **params})
        # categorical columns appear as strings in the signature (MLflow schemas have no category type)
        X_sig = X_test.head(200).apply(lambda c: c.astype(str) if isinstance(c.dtype, pd.CategoricalDtype) else c)
        sig = infer_signature(X_sig, p_test[:200])
        if flavor == "sklearn":
            import skops.io  # MLflow saves scikit-learn models with skops; trust the plain numpy/sklearn types in them

            trusted = skops.io.get_untrusted_types(data=skops.io.dumps(model))
            info = mlflow.sklearn.log_model(model, name="model", signature=sig, skops_trusted_types=trusted, **kw)
        elif flavor == "lightgbm":
            info = mlflow.lightgbm.log_model(model, name="model", signature=sig, **kw)
        elif flavor == "xgboost":
            info = mlflow.xgboost.log_model(model, name="model", signature=sig, **kw)
        else:
            info = mlflow.pyfunc.log_model(name="model", python_model=model, signature=sig, **kw)
        m = metrics(y_test, p_test, thr)
        mlflow.log_metrics(
            {f"test_{k}": v for k, v in m.items() if k not in ("n", "threshold", "on_time_rate")},
            model_id=info.model_id,
        )
        cm_fig, roc_fig = _figures(y_test, p_test, thr, title)
        mlflow.log_figure(cm_fig, "plots/confusion_matrix.png")
        mlflow.log_figure(roc_fig, "plots/roc_curve.png")
        plt.close("all")
        mv = mlflow.register_model(info.model_uri, REGISTRY)
        client = MlflowClient()
        client.update_model_version(REGISTRY, mv.version, description=f"{title}. {why}")
        client.set_model_version_tag(REGISTRY, mv.version, "test_accuracy", f"{m['accuracy']:.4f}")
        client.set_model_version_tag(REGISTRY, mv.version, "test_roc_auc", f"{m['roc_auc']:.4f}")
        print(
            f"  v{mv.version:<3} {title:<48} acc {m['accuracy']:.4f}  AUC {m['roc_auc']:.4f}  (run {run.info.run_id[:8]})"
        )
        return mv.version, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracking-uri", default=f"sqlite:///{MODULE_ROOT / 'mlflow.db'}")
    ap.add_argument("--samples-cache", default=None, help="pickle of build_dataset() output (skips the rebuild)")
    a = ap.parse_args()
    t0 = time.time()
    mlflow.set_tracking_uri(a.tracking_uri)
    mlflow.set_experiment(EXPERIMENT)

    res = json.loads((REPORTS_DIR / "oulad_results.json").read_text())
    card = json.loads((MODELS_DIR / "model_card.json").read_text())
    thr = float(card["threshold"])
    tun = res["tuning"]

    if a.samples_cache:
        df = pd.read_pickle(a.samples_cache)
    else:
        from .data_oulad import load_oulad
        from .features_oulad import build_dataset

        print("Building the 663k OULAD samples from 07-datasets (a few minutes)...")
        df = build_dataset(load_oulad())
    tr = df[df.code_presentation.isin(TRAIN_PRESENTATIONS)]
    te = df[df.code_presentation == TEST_PRESENTATION]
    yt = te.y.to_numpy()
    print(f"[{time.time() - t0:5.0f}s] samples: train {len(tr):,} | test {len(te):,}")
    data_params = {
        "train_cohorts": ",".join(TRAIN_PRESENTATIONS),
        "test_cohort": TEST_PRESENTATION,
        "train_samples": len(tr),
        "test_samples": len(te),
    }

    lgbm_file = str(MODELS_DIR / card["lightgbm_file"])
    xgb_file = str(MODELS_DIR / card["xgboost_file"])
    xgb_final = xgb.XGBClassifier()
    xgb_final.load_model(xgb_file)
    lgbm_tuned = lgb.Booster(model_file=lgbm_file)
    p_x = xgb_final.predict_proba(te[F])[:, 1]
    p_l = lgbm_tuned.predict(te[F])
    nl = int(tun["lightgbm"]["trees"] * REFIT_FACTOR)

    with mlflow.start_run(run_name="model-history (every model up to the final)") as parent:
        mlflow.set_tags(
            {
                "kind": "model history",
                "registry": REGISTRY,
                "mlflow.note.content": (
                    "Every model built while developing the M3 task-completion predictor, in the order it was "
                    "built. All are evaluated on the same held-out 2014J cohort. Model choice was made on the "
                    "2014B validation cohort, not on these test numbers. Each child run holds the loadable model, "
                    "its settings, metrics, confusion matrix and ROC curve, and is a numbered version of the "
                    f"'{REGISTRY}' registered model. The final model is the last version, alias 'champion'."
                ),
            }
        )
        mlflow.log_params(data_params)
        print("Logging models (version · model · test accuracy · test ROC-AUC):")
        s = 0

        def step():
            nonlocal s
            s += 1
            return s

        majority = DummyClassifier(strategy="most_frequent").fit(tr[F], tr.y)
        _log(
            step(),
            "majority-class",
            "Baseline 1: always predict the most common class",
            "Lower bound: the accuracy you get without learning anything.",
            "sklearn",
            majority,
            te[F],
            yt,
            majority.predict_proba(te[F])[:, 1],
            {"algorithm": "DummyClassifier(most_frequent)", **data_params},
            extra_tags={"group": "baseline"},
        )

        _log(
            step(),
            "simple-rule",
            "Baseline 2: simple rule (past on-time rate)",
            "A hand-written rule a person could use; any ML model must beat this to be worth it.",
            "pyfunc",
            RuleModel(),
            te[F],
            yt,
            rule_baseline(te),
            {"algorithm": "rule: prior on-time rate, else recent-click percentile", **data_params},
            extra_tags={"group": "baseline"},
        )

        lr = logreg(tr[F], tr.y)
        _log(
            step(),
            "logistic-regression",
            "Baseline 3: logistic regression",
            "Standard linear ML baseline (median imputation, scaling, one-hot encoding).",
            "sklearn",
            lr,
            te[F],
            yt,
            lr.predict_proba(te[F])[:, 1],
            {"algorithm": "LogisticRegression(C=1.0)", "n_features": len(F), **data_params},
            extra_tags={"group": "baseline"},
        )

        lv1 = lgbm_fit({}, tr[F_V1], tr.y, n_estimators=300)
        _log(
            step(),
            "lightgbm-default-features-v1",
            "LightGBM, default settings, feature set v1",
            f"First gradient-boosting model, on the first feature set ({len(F_V1)} features, before the "
            "EXTRA_V2 engagement-trend features were added).",
            "lightgbm",
            lv1,
            te[F_V1],
            yt,
            lv1.predict_proba(te[F_V1])[:, 1],
            {"algorithm": "LightGBM", "n_estimators": 300, "n_features": len(F_V1), "feature_set": "v1", **data_params},
            extra_tags={"group": "development"},
        )

        lv2 = lgbm_fit({}, tr[F], tr.y, n_estimators=300)
        _log(
            step(),
            "lightgbm-default-features-v2",
            "LightGBM, default settings, feature set v2",
            f"Same model after adding the {len(EXTRA_V2)} EXTRA_V2 features (validation AUC 0.9316 -> 0.9325). "
            "This is the feature set used from here on.",
            "lightgbm",
            lv2,
            te[F],
            yt,
            lv2.predict_proba(te[F])[:, 1],
            {
                "algorithm": "LightGBM",
                "n_estimators": 300,
                "n_features": len(F),
                "feature_set": "v2 (behaviour)",
                **data_params,
            },
            extra_tags={"group": "development"},
        )

        _log(
            step(),
            "lightgbm-tuned",
            "LightGBM, tuned with Optuna",
            f"Best of {tun['lightgbm']['trials']} Optuna trials (validation AUC {tun['lightgbm']['val_auc']:.4f}). "
            "Loaded from the saved model file.",
            "lightgbm",
            lgbm_tuned,
            te[F],
            yt,
            p_l,
            {
                "algorithm": "LightGBM",
                "n_estimators": nl,
                "n_features": len(F),
                **{f"lgbm_{k}": v for k, v in tun["lightgbm"]["params"].items()},
                **data_params,
            },
            extra_tags={"group": "candidate"},
        )

        _log(
            step(),
            "lgbm-xgb-average",
            "Average of tuned LightGBM and XGBoost",
            f"Ensemble candidate. Validation AUC {tun['validation_auc_by_candidate']['ensemble']:.4f}, "
            f"below XGBoost alone ({tun['validation_auc_by_candidate']['xgboost']:.4f}), so it was not chosen.",
            "pyfunc",
            AverageModel(tr.code_module.cat.categories),
            te[F],
            yt,
            (p_l + p_x) / 2,
            {"algorithm": "mean(LightGBM, XGBoost)", **data_params},
            extra_tags={"group": "candidate"},
            artifacts={"lightgbm": lgbm_file, "xgboost": xgb_file},
        )

        for name in ("behaviour+demographics", "behaviour+scores", "all"):
            fs = FEATURE_SETS[name]
            m = lgbm_fit(tun["lightgbm"]["params"], tr[fs], tr.y, n_estimators=nl)
            _log(
                step(),
                f"ablation-{name.replace('+', '-plus-')}",
                f"Ablation: tuned LightGBM + {name}",
                "Checks whether demographics or earlier marks add accuracy. They add at most +0.001 AUC, so the "
                "final model leaves them out (fairer, and the app doesn't collect them).",
                "lightgbm",
                m,
                te[fs],
                yt,
                m.predict_proba(te[fs])[:, 1],
                {"algorithm": "LightGBM (tuned settings)", "feature_set": name, "n_features": len(fs), **data_params},
                extra_tags={"group": "ablation"},
            )

        v, m = _log(
            step(),
            "FINAL-xgboost-tuned",
            "FINAL MODEL: XGBoost, tuned with Optuna",
            f"Chosen on the 2014B validation cohort (best validation AUC "
            f"{tun['validation_auc_by_candidate']['xgboost']:.4f}). Metrics use threshold 0.5 like every "
            f"other model; the validation-tuned threshold {thr:.3f} gives "
            f"{metrics(yt, p_x, thr)['accuracy']:.4f} accuracy. This is the model used by Module 3.",
            "xgboost",
            xgb_final,
            te[F],
            yt,
            p_x,
            {
                "algorithm": "XGBoost",
                "n_estimators": xgb_final.get_booster().num_boosted_rounds(),
                "n_features": len(F),
                **{f"xgb_{k}": v for k, v in tun["xgboost"]["params"].items()},
                "validation_tuned_threshold": round(thr, 3),
                **data_params,
            },
            extra_tags={"group": "final", "final_model": "true"},
        )
        client = MlflowClient()
        client.set_registered_model_alias(REGISTRY, "champion", v)
        client.update_registered_model(
            REGISTRY,
            description="Module 3 task-completion predictor (OULAD). Versions are every model built in order; "
            "the alias 'champion' marks the final model.",
        )
        mlflow.log_metrics({"final_test_accuracy": m["accuracy"], "final_test_roc_auc": m["roc_auc"]})
        print(
            f"[{time.time() - t0:5.0f}s] done: {s} models in run {parent.info.run_id[:8]}; "
            f"registry '{REGISTRY}' v{v} = champion"
        )
    print(f"View: mlflow ui --backend-store-uri {a.tracking_uri}   then open http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
