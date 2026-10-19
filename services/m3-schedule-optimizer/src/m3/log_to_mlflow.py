"""Record the trained OULAD completion model in MLflow — a browsable, permanent record (proof) of how it was built.

Nothing is retrained: this reads the saved outputs of `m3.train_oulad` (results JSON, model card, Optuna history,
figures, model file) and logs them as MLflow runs:

  oulad-completion-model (parent run)         dataset, split, features, final settings, all test metrics, charts, model
  ├── tuning/lgbm-trial-N, tuning/xgb-trial-N  every hyper-parameter trial with its validation AUC
  └── baseline/<name>                          each comparison model's test metrics

Usage (from services/m3-schedule-optimizer):
    PYTHONPATH=src python -m m3.log_to_mlflow
    mlflow ui --backend-store-uri sqlite:///mlflow.db      # then open http://127.0.0.1:5000
"""

import argparse
import json

import mlflow
import optuna

from .config import MODELS_DIR, MODULE_ROOT, REPORTS_DIR

EXPERIMENT = "M3 - Task completion prediction (OULAD)"
NICE = {
    "majority_class": "always-not-on-time",
    "rule_baseline": "simple-rule",
    "logistic_regression": "logistic-regression",
    "lightgbm_untuned": "lightgbm-default",
    "lightgbm_tuned": "lightgbm-tuned",
    "xgboost_tuned": "xgboost-tuned",
    "ensemble_tuned": "lgbm-xgb-average",
}
METRIC_KEYS = [
    "accuracy",
    "roc_auc",
    "macro_f1",
    "balanced_accuracy",
    "f1_at_risk",
    "recall_at_risk",
    "precision_at_risk",
    "f1_on_time",
    "pr_auc_on_time",
    "brier",
    "log_loss",
    "ece",
]


def _metrics(m, prefix=""):
    return {f"{prefix}{k}": float(m[k]) for k in METRIC_KEYS if k in m}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracking-uri", default=f"sqlite:///{MODULE_ROOT / 'mlflow.db'}")
    a = ap.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    mlflow.set_tracking_uri(a.tracking_uri)
    mlflow.set_experiment(EXPERIMENT)

    res = json.loads((REPORTS_DIR / "oulad_results.json").read_text())
    card = json.loads((MODELS_DIR / "model_card.json").read_text())
    chosen = res["tuning"]["chosen"]
    final_key = f"FINAL_{chosen}@0.5"

    with mlflow.start_run(run_name="oulad-completion-model") as parent:
        mlflow.set_tags(
            {
                "module": "M3 Predictive Adaptive Schedule Optimization",
                "owner": "G.W.D.D.N. Kumarasinghe (IT23267336)",
                "project": "J26-SE-360",
                "final_model": chosen,
                "task": "binary classification: task completed on time?",
                "mlflow.note.content": (
                    "Predicts whether a task will be submitted by its deadline, 1/3/7/14 days before the deadline, using only "
                    "information available at that time. Trained on OULAD (Kuzilek et al., 2017, CC BY 4.0) with a temporal "
                    "split: train 2013B-2014B, validate 2014B (tuning/model choice), test 2014J (held out). "
                    f"Test accuracy {res['test'][final_key]['accuracy']:.4f}, ROC-AUC {res['test'][final_key]['roc_auc']:.4f}."
                ),
            }
        )
        mlflow.log_params(
            {
                "dataset": "OULAD (Open University Learning Analytics Dataset)",
                "dataset_doi": "10.1038/sdata.2017.171",
                "label": card["label"],
                "prediction_time": card["prediction_time"],
                "train_cohorts": ",".join(res["split"]["final_train"]),
                "validation_cohort": res["split"]["validation"],
                "test_cohort": res["split"]["test"],
                "n_features": len(res["features"]),
                "feature_set": res["feature_set"],
                "final_model": chosen,
                "decision_threshold": round(res["tuning"]["threshold"], 3),
                **{f"samples_{k}": v for k, v in res["sizes"].items()},
                **{f"{chosen}_{k}": v for k, v in res["tuning"][chosen]["params"].items()},
                f"{chosen}_trees": res["tuning"][chosen]["trees"],
            }
        )
        mlflow.log_metrics(_metrics(res["test"][final_key], "test_"))
        mlflow.log_metrics({f"validation_auc_{k}": v for k, v in res["tuning"]["validation_auc_by_candidate"].items()})
        for h, m in res["test_by_horizon"].items():
            mlflow.log_metrics({f"test_accuracy_H{h}": m["accuracy"], f"test_auc_H{h}": m["roc_auc"]})
        for t, m in res["test_by_type"].items():
            mlflow.log_metrics({f"test_accuracy_{t}": m["accuracy"], f"test_auc_{t}": m["roc_auc"]})
        for name, m in res["ablation_feature_sets_lightgbm"].items():
            mlflow.log_metric(f"ablation_auc_{name.replace('+', '_plus_')}", m["roc_auc"])
        mlflow.log_dict({"features": res["features"]}, "inputs/features.json")
        mlflow.log_dict(res["shap_mean_abs_lightgbm"], "explainability/shap_mean_abs.json")
        for f in [*sorted(REPORTS_DIR.glob("*.png")), REPORTS_DIR / "README.md", REPORTS_DIR / "oulad_results.json"]:
            if f.exists():
                mlflow.log_artifact(str(f), "reports")
        mlflow.log_artifact(str(MODELS_DIR / "model_card.json"), "model")
        model_file = MODELS_DIR / card[f"{chosen}_file"]
        if model_file.exists():
            mlflow.log_artifact(str(model_file), "model")
        nb = MODULE_ROOT / "notebooks" / "01_completion_model_evidence.ipynb"
        if nb.exists():
            mlflow.log_artifact(str(nb), "evidence")

        # every comparison model on the same test set
        for k, label in NICE.items():
            if k in res["test"]:
                with mlflow.start_run(run_name=f"baseline/{label}", nested=True):
                    mlflow.set_tag("kind", "test comparison")
                    mlflow.log_param("model", label)
                    mlflow.log_metrics(_metrics(res["test"][k], "test_"))

        # every hyper-parameter tuning trial
        db = MODELS_DIR / "optuna_tuning.db"
        n_trials = 0
        if db.exists():
            for study in ("oulad_lgbm", "oulad_xgb"):
                st = optuna.load_study(study_name=study, storage=f"sqlite:///{db}")
                for tr in st.trials:
                    if tr.state.name != "COMPLETE":
                        continue
                    algo = study.replace("oulad_", "")
                    with mlflow.start_run(run_name=f"tuning/{algo}-trial-{tr.number}", nested=True):
                        mlflow.set_tag("kind", "hyper-parameter trial")
                        mlflow.log_params(
                            {"algorithm": algo, **tr.params, "trees": tr.user_attrs.get("best_iteration")}
                        )
                        mlflow.log_metric("validation_auc", tr.value)
                    n_trials += 1
        print(f"Logged run {parent.info.run_id}: final {chosen} + {len(NICE)} comparisons + {n_trials} tuning trials")
        print(f"View it:  mlflow ui --backend-store-uri {a.tracking_uri}   then open http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
