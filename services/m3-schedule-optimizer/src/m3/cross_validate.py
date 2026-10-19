"""Cross-validation of the final model settings, logged to MLflow.

Shows whether the result is stable or one lucky test semester. It uses two leakage-safe kinds of
cross-validation, because a random k-fold split would put the same student and the same task in both
training and test:

  1. Rolling-origin (time) CV: always learn from earlier semesters and test on the next one.
       fold 1: 2013B                -> 2013J
       fold 2: 2013B + 2013J        -> 2014B
       fold 3: 2013B + 2013J + 2014B -> 2014J   (= the reported final test)
  2. Student-grouped 5-fold CV on the training semesters (2013B + 2013J + 2014B): every student's samples
     sit in exactly one fold, so the model is never tested on a student it learned from.

Every fold uses the final XGBoost settings, the same number of trees and threshold 0.5. Results are summarised
as mean +/- standard deviation.

Usage (from services/m3-schedule-optimizer):
    PYTHONPATH=src python -m m3.cross_validate
    PYTHONPATH=src python -m m3.cross_validate --samples-cache path/to/oulad_samples.pkl
"""
# ruff: noqa: N806  -- ML convention: F (feature list) is uppercase

import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from .config import MODULE_ROOT, REPORTS_DIR, SEED  # noqa: E402
from .features_oulad import FEATURE_SETS  # noqa: E402
from .train_oulad import REFIT_FACTOR, metrics, xgb_fit  # noqa: E402

EXPERIMENT = "M3 - Task completion prediction (OULAD)"
F = FEATURE_SETS["behaviour"]
KEYS = ["accuracy", "roc_auc", "macro_f1", "recall_at_risk", "precision_at_risk", "brier", "ece"]
ROLLING = [
    (["2013B"], "2013J"),
    (["2013B", "2013J"], "2014B"),
    (["2013B", "2013J", "2014B"], "2014J"),
]


def _summary(rows):
    df = pd.DataFrame(rows)
    return {f"{k}_{s}": float(getattr(df[k], s)()) for k in KEYS for s in ("mean", "std")}


def _chart(rows, title, labels):
    acc = [r["accuracy"] * 100 for r in rows]
    auc = [r["roc_auc"] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, vals, name, fmt in ((a1, acc, "Accuracy (%)", "{:.1f}"), (a2, auc, "ROC-AUC", "{:.3f}")):
        bars = ax.bar(labels, vals, color="#4C78A8")
        m = float(np.mean(vals))
        ax.axhline(
            m, color="#E45756", ls="--", lw=1.5, label=f"mean {fmt.format(m)} ± {fmt.format(np.std(vals, ddof=1))}"
        )
        ax.bar_label(bars, [fmt.format(v) for v in vals], fontsize=8)
        lo = min(vals)
        ax.set_ylim(lo - (3 if "%" in name else 0.03), max(vals) + (2 if "%" in name else 0.02))
        ax.set_title(name)
        ax.legend(fontsize=8, loc="lower right")
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracking-uri", default=f"sqlite:///{MODULE_ROOT / 'mlflow.db'}")
    ap.add_argument("--samples-cache", default=None, help="pickle of build_dataset() output (skips the rebuild)")
    ap.add_argument("--folds", type=int, default=5, help="number of student-grouped folds")
    a = ap.parse_args()
    t0 = time.time()
    mlflow.set_tracking_uri(a.tracking_uri)
    mlflow.set_experiment(EXPERIMENT)

    res = json.loads((REPORTS_DIR / "oulad_results.json").read_text())
    params = res["tuning"]["xgboost"]["params"]
    trees = int(res["tuning"]["xgboost"]["trees"] * REFIT_FACTOR)

    if a.samples_cache:
        df = pd.read_pickle(a.samples_cache)
    else:
        from .data_oulad import load_oulad
        from .features_oulad import build_dataset

        print("Building the 663k OULAD samples from 07-datasets (a few minutes)...")
        df = build_dataset(load_oulad())
    print(f"[{time.time() - t0:5.0f}s] {len(df):,} samples loaded")

    def fit_eval(tr, te):
        m = xgb_fit(params, tr[F], tr.y, n_estimators=trees)
        return metrics(te.y.to_numpy(), m.predict_proba(te[F])[:, 1], 0.5)

    with mlflow.start_run(run_name="cross-validation (final XGBoost settings)") as parent:
        mlflow.set_tags(
            {
                "kind": "cross-validation",
                "mlflow.note.content": (
                    "Is 83.4% stable or one lucky semester? Two leakage-safe cross-validations with the final "
                    "XGBoost settings. (1) Rolling-origin: always learn from earlier semesters and test on the "
                    "next one, like real use. (2) Student-grouped 5-fold on 2013B+2013J+2014B: no student is in "
                    "both training and test. Random k-fold is not used because it would test on students and "
                    "tasks the model has already seen. Metrics use threshold 0.5. Note: time fold 2 tests on 2014B, the "
                    "semester used for tuning, so it is slightly optimistic; time fold 3 is the untouched final test."
                ),
            }
        )
        mlflow.log_params(
            {
                "model": "XGBoost (final settings)",
                "n_estimators": trees,
                "n_features": len(F),
                "grouped_folds": a.folds,
                **{f"xgb_{k}": v for k, v in params.items()},
            }
        )

        # ---- 1. rolling-origin (time) CV
        roll, labels = [], []
        for i, (train_sems, test_sem) in enumerate(ROLLING, 1):
            tr = df[df.code_presentation.isin(train_sems)]
            te = df[df.code_presentation == test_sem]
            m = fit_eval(tr, te)
            roll.append(m)
            labels.append(f"{'+'.join(train_sems)}\n-> {test_sem}")
            with mlflow.start_run(run_name=f"time-fold-{i}: {'+'.join(train_sems)} -> {test_sem}", nested=True):
                mlflow.set_tags({"kind": "cv fold", "cv": "rolling-origin"})
                mlflow.log_params(
                    {
                        "train_semesters": ",".join(train_sems),
                        "test_semester": test_sem,
                        "train_samples": len(tr),
                        "test_samples": len(te),
                    }
                )
                mlflow.log_metrics({k: m[k] for k in KEYS})
            print(
                f"[{time.time() - t0:5.0f}s] time fold {i}: {'+'.join(train_sems)} -> {test_sem}  "
                f"acc {m['accuracy']:.4f}  AUC {m['roc_auc']:.4f}"
            )
        s_roll = _summary(roll)
        mlflow.log_metrics({f"time_cv_{k}": v for k, v in s_roll.items()})
        mlflow.log_figure(_chart(roll, "Rolling-origin (time) cross-validation", labels), "cv/time_cv.png")

        # ---- 2. student-grouped k-fold CV on the training semesters
        pool = df[df.code_presentation.isin(ROLLING[-1][0])].reset_index(drop=True)
        rng = np.random.default_rng(SEED)
        students = pool.id_student.unique()
        group_of = dict(zip(students, rng.permutation(len(students)), strict=True))  # shuffle students
        groups = pool.id_student.map(group_of).to_numpy()
        grp, glabels = [], []
        for i, (tri, tei) in enumerate(GroupKFold(n_splits=a.folds).split(pool, groups=groups), 1):
            tr, te = pool.iloc[tri], pool.iloc[tei]
            assert not set(tr.id_student) & set(te.id_student)  # no student in both
            m = fit_eval(tr, te)
            grp.append(m)
            glabels.append(f"fold {i}")
            with mlflow.start_run(run_name=f"student-fold-{i}", nested=True):
                mlflow.set_tags({"kind": "cv fold", "cv": "student-grouped"})
                mlflow.log_params(
                    {
                        "train_samples": len(tr),
                        "test_samples": len(te),
                        "train_students": tr.id_student.nunique(),
                        "test_students": te.id_student.nunique(),
                    }
                )
                mlflow.log_metrics({k: m[k] for k in KEYS})
            print(
                f"[{time.time() - t0:5.0f}s] student fold {i}/{a.folds}: acc {m['accuracy']:.4f}  AUC {m['roc_auc']:.4f}"
            )
        s_grp = _summary(grp)
        mlflow.log_metrics({f"student_cv_{k}": v for k, v in s_grp.items()})
        mlflow.log_figure(_chart(grp, f"Student-grouped {a.folds}-fold cross-validation", glabels), "cv/student_cv.png")

        table = pd.DataFrame(
            [
                {"cv": "rolling-origin", "fold": lab.replace("\n", " "), **{k: r[k] for k in KEYS}}
                for lab, r in zip(labels, roll, strict=True)
            ]
            + [
                {"cv": "student-grouped", "fold": lab, **{k: r[k] for k in KEYS}}
                for lab, r in zip(glabels, grp, strict=True)
            ]
        )
        mlflow.log_table(table, "cv/folds.json")
        mlflow.log_text(table.round(4).to_csv(index=False), "cv/folds.csv")
        print(
            f"[{time.time() - t0:5.0f}s] done (run {parent.info.run_id[:8]})\n"
            f"  time CV     accuracy {s_roll['accuracy_mean']:.4f} ± {s_roll['accuracy_std']:.4f} | "
            f"AUC {s_roll['roc_auc_mean']:.4f} ± {s_roll['roc_auc_std']:.4f}\n"
            f"  student CV  accuracy {s_grp['accuracy_mean']:.4f} ± {s_grp['accuracy_std']:.4f} | "
            f"AUC {s_grp['roc_auc_mean']:.4f} ± {s_grp['roc_auc_std']:.4f}"
        )


if __name__ == "__main__":
    main()
