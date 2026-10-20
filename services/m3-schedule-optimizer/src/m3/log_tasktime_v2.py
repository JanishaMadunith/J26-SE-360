"""Log round 2 of the time-of-day model (m3.tasktime_v2) to MLflow from its saved results.

One parent run "tasktime-round2" in the experiment "M3 - Task time-of-day (MS-LaTTE)" with one child run per model
(v1-v8), so they can be compared side by side in the MLflow UI. Re-running replaces nothing: it adds a new parent.

Usage (from services/m3-schedule-optimizer):
    PYTHONPATH=src python -m m3.log_tasktime_v2
"""

import argparse
import json
from pathlib import Path

from .config import MODULE_ROOT, REPORTS_DIR

EXPERIMENT = "M3 - Task time-of-day (MS-LaTTE)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracking-uri", default=f"sqlite:///{MODULE_ROOT / 'mlflow.db'}")
    ap.add_argument("--results", default=str(REPORTS_DIR / "tasktime_v2"))
    a = ap.parse_args()
    import mlflow

    folder = Path(a.results)
    res = json.loads((folder / "results.json").read_text())
    best = max(res["models"], key=lambda m: m["val_accuracy"] or 0)  # chosen on validation, not on test
    mlflow.set_tracking_uri(a.tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="tasktime-round2"):
        mlflow.set_tags(
            {
                "kind": "model comparison",
                "dataset": "MS-LaTTE",
                "chosen": best["name"],
                "mlflow.note.content": (
                    "Round 2: can a language model beat TF-IDF? 8 models on the same test split (2,021 tasks). "
                    f"Best on validation: {best['name']} - test accuracy {best['accuracy']:.1%}, "
                    f"top-2 {best['top2_accuracy']:.1%}. People agree with each other only "
                    f"{res['baselines']['one_person_vs_others']:.1%} of the time, so ~60% is close to the ceiling."
                ),
            }
        )
        mlflow.log_params({"split": res["split"], "n_test": res["n_test"], "chosen": best["key"]})
        mlflow.log_metrics(
            {
                "best_test_accuracy": best["accuracy"],
                "best_test_top2_accuracy": best["top2_accuracy"],
                "baseline_most_common": res["baselines"]["always_most_common"],
                "baseline_one_person_vs_others": res["baselines"]["one_person_vs_others"],
            }
        )
        for f in ("results.json", "accuracy.png"):
            if (folder / f).exists():
                mlflow.log_artifact(str(folder / f))
        for m in res["models"]:
            with mlflow.start_run(run_name=m["name"], nested=True):
                mlflow.set_tags({"kind": "round 2 model", "chosen": str(m["key"] == best["key"])})
                params = {"features": m["features"], "labels": m["labels"]}
                for k in ("C", "epochs", "lr", "batch"):
                    if k in m:
                        params[k] = m[k]
                if "weights" in m:
                    params["ensemble_weights"] = json.dumps(m["weights"])
                mlflow.log_params(params)
                mlflow.log_metrics(
                    {
                        k: m[k]
                        for k in ("accuracy", "top2_accuracy", "macro_f1", "annotator_agreement", "val_accuracy")
                        if m.get(k) is not None
                    }
                )
                for step, acc in enumerate(m.get("val_curve", []), 1):
                    mlflow.log_metric("val_accuracy_per_epoch", acc, step=step)
    print(f"logged {len(res['models'])} models; chosen: {best['name']} ({best['accuracy']:.1%})")


if __name__ == "__main__":
    main()
