"""Train and evaluate the task-completion model on OULAD.

Protocol (no test-set peeking):
  1. Tune LightGBM and XGBoost with Optuna: train on 2013B + 2013J, validate on 2014B (ROC-AUC).
  2. Pick the final model (LightGBM, XGBoost or their average) and the decision threshold on 2014B.
  3. Refit on 2013B + 2013J + 2014B with the tuned settings and a fixed number of trees.
  4. Evaluate ONCE on 2014J, which was never used for training, tuning or model selection.

Usage:
    DATA_DIR=/path/to/07-datasets python -m m3.train_oulad --lgbm-trials 35 --xgb-trials 20
"""
# ruff: noqa: N803, N806  -- ML convention: X (features), F (feature list), H (horizon) are uppercase

import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import MODELS_DIR, REPORTS_DIR, SEED, TEST_PRESENTATION, TRAIN_PRESENTATIONS, VALID_PRESENTATION
from .data_oulad import load_oulad
from .features_oulad import CATEGORICAL, FEATURE_SETS, build_dataset

MAIN_SET = "behaviour"
REFIT_FACTOR = 1.15  # more trees when refitting on ~1.5x more data


# ----------------------------------------------------------------------------- metrics
def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(
        sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())
    )


def metrics(y, p, thr=0.5):
    y = np.asarray(y)
    p = np.asarray(p)
    yhat = (p >= thr).astype(int)
    return {
        "n": len(y),
        "on_time_rate": float(y.mean()),
        "threshold": float(thr),
        "accuracy": float(accuracy_score(y, yhat)),
        "balanced_accuracy": float(balanced_accuracy_score(y, yhat)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc_on_time": float(average_precision_score(y, p)),
        "f1_on_time": float(f1_score(y, yhat)),
        "f1_at_risk": float(f1_score(1 - y, 1 - yhat)),
        "macro_f1": float(f1_score(y, yhat, average="macro")),
        "precision_at_risk": float(precision_score(1 - y, 1 - yhat, zero_division=0)),
        "recall_at_risk": float(recall_score(1 - y, 1 - yhat)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
        "ece": ece(y, p),
    }


def best_threshold(y, p):
    grid = np.linspace(0.2, 0.8, 121)
    acc = [accuracy_score(y, p >= g) for g in grid]
    return float(grid[int(np.argmax(acc))])


# ----------------------------------------------------------------------------- models
def lgbm_fit(params, Xtr, ytr, Xva=None, yva=None, n_estimators=5000):
    m = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=n_estimators,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
        bagging_freq=1,
        **params,
    )
    if Xva is not None:
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(200, verbose=False)])
    else:
        m.fit(Xtr, ytr)
    return m


def xgb_fit(params, Xtr, ytr, Xva=None, yva=None, n_estimators=5000):
    kw = dict(
        objective="binary:logistic",
        n_estimators=n_estimators,
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED,
        n_jobs=-1,
        eval_metric="auc",
        **params,
    )
    if Xva is not None:
        m = xgb.XGBClassifier(early_stopping_rounds=200, **kw)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    else:
        m = xgb.XGBClassifier(**kw)
        m.fit(Xtr, ytr, verbose=False)
    return m


def lgbm_space(tr):
    return {
        "learning_rate": tr.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": tr.suggest_int("num_leaves", 15, 255, log=True),
        "min_child_samples": tr.suggest_int("min_child_samples", 20, 500, log=True),
        "colsample_bytree": tr.suggest_float("colsample_bytree", 0.4, 1.0),
        "subsample": tr.suggest_float("subsample", 0.5, 1.0),
        "reg_alpha": tr.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": tr.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_split_gain": tr.suggest_float("min_split_gain", 0.0, 1.0),
    }


def xgb_space(tr):
    return {
        "learning_rate": tr.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": tr.suggest_int("max_depth", 3, 10),
        "min_child_weight": tr.suggest_float("min_child_weight", 1.0, 100.0, log=True),
        "subsample": tr.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": tr.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": tr.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": tr.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "gamma": tr.suggest_float("gamma", 0.0, 5.0),
        "max_cat_to_onehot": 8,
    }


def tune(kind, Xtr, ytr, Xva, yva, n_trials, timeout, storage=None):
    """Optuna search. With `storage` (an sqlite URL) every finished trial is saved, so a crashed or
    restarted run resumes where it stopped instead of starting over."""
    fit, space = (lgbm_fit, lgbm_space) if kind == "lgbm" else (xgb_fit, xgb_space)

    def objective(trial):
        params = space(trial)
        m = fit(params, Xtr, ytr, Xva, yva)
        p = m.predict_proba(Xva)[:, 1]
        it = m.best_iteration_ if kind == "lgbm" else m.best_iteration + 1
        trial.set_user_attr("best_iteration", int(it))
        return roc_auc_score(yva, p)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        study_name=f"oulad_{kind}",
        storage=storage,
        load_if_exists=True,
    )
    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if done < n_trials:
        study.optimize(objective, n_trials=n_trials - done, timeout=timeout, show_progress_bar=False)
    best = study.best_trial
    return best.params, best.user_attrs["best_iteration"], best.value, len(study.trials)


def logreg(Xtr, ytr):
    num = [c for c in Xtr.columns if c not in CATEGORICAL]
    cat = [c for c in Xtr.columns if c in CATEGORICAL]
    pre = ColumnTransformer(
        [
            ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ]
    )
    m = make_pipeline(pre, LogisticRegression(max_iter=3000, C=1.0))
    m.fit(Xtr, ytr)
    return m


def rule_baseline(X):
    """Simple rule: on time if the student was on time for most earlier tasks; if no history,
    on time if they are in the top half of their cohort for clicks in the last 14 days."""
    r = X["prior_on_time_rate"]
    return np.where(r.notna(), r.fillna(0), X["pct_clicks_14"]).astype(float)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lgbm-trials", type=int, default=35)
    ap.add_argument("--xgb-trials", type=int, default=20)
    ap.add_argument("--timeout-min", type=float, default=25)
    ap.add_argument("--samples-cache", default=None, help="optional pickle of build_dataset() output")
    a = ap.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def log(*x):
        print(f"[{time.time() - t0:7.1f}s]", *x, flush=True)

    df = pd.read_pickle(a.samples_cache) if a.samples_cache else build_dataset(load_oulad())
    F = FEATURE_SETS[MAIN_SET]
    tune_tr = df[df.code_presentation.isin([p for p in TRAIN_PRESENTATIONS if p != VALID_PRESENTATION])]
    val = df[df.code_presentation == VALID_PRESENTATION]
    full_tr = df[df.code_presentation.isin(TRAIN_PRESENTATIONS)]
    test = df[df.code_presentation == TEST_PRESENTATION]
    log(f"samples: tune-train {len(tune_tr):,} | val {len(val):,} | full-train {len(full_tr):,} | test {len(test):,}")
    out = {
        "protocol": __doc__.strip().splitlines()[2:7],
        "feature_set": MAIN_SET,
        "features": F,
        "split": {
            "tune_train": [p for p in TRAIN_PRESENTATIONS if p != VALID_PRESENTATION],
            "validation": VALID_PRESENTATION,
            "final_train": TRAIN_PRESENTATIONS,
            "test": TEST_PRESENTATION,
        },
        "sizes": {"tune_train": len(tune_tr), "validation": len(val), "final_train": len(full_tr), "test": len(test)},
    }

    # ---- 1. tuning on validation
    storage = f"sqlite:///{MODELS_DIR / 'optuna_tuning.db'}"
    lp, lit, lauc, ln = tune("lgbm", tune_tr[F], tune_tr.y, val[F], val.y, a.lgbm_trials, a.timeout_min * 60, storage)
    log(f"LightGBM tuned: val AUC {lauc:.4f} ({ln} trials, {lit} trees)")
    xp, xit, xauc, xn = tune("xgb", tune_tr[F], tune_tr.y, val[F], val.y, a.xgb_trials, a.timeout_min * 60, storage)
    log(f"XGBoost tuned:  val AUC {xauc:.4f} ({xn} trials, {xit} trees)")

    # ---- 2. model + threshold selection on validation
    lv = lgbm_fit(lp, tune_tr[F], tune_tr.y, n_estimators=lit).predict_proba(val[F])[:, 1]
    xv = xgb_fit(xp, tune_tr[F], tune_tr.y, n_estimators=xit).predict_proba(val[F])[:, 1]
    cands = {"lightgbm": lv, "xgboost": xv, "ensemble": (lv + xv) / 2}
    val_auc = {k: float(roc_auc_score(val.y, v)) for k, v in cands.items()}
    choice = max(val_auc, key=val_auc.get)
    thr = best_threshold(val.y.to_numpy(), cands[choice])
    log(f"validation AUCs {val_auc} -> final model: {choice}, threshold {thr:.3f}")
    out["tuning"] = {
        "lightgbm": {"params": lp, "trees": lit, "val_auc": lauc, "trials": ln},
        "xgboost": {"params": xp, "trees": xit, "val_auc": xauc, "trials": xn},
        "validation_auc_by_candidate": val_auc,
        "chosen": choice,
        "threshold": thr,
        "validation_metrics_chosen": metrics(val.y, cands[choice], thr),
    }

    # ---- 3. refit on all training cohorts
    nl, nx = int(lit * REFIT_FACTOR), int(xit * REFIT_FACTOR)
    lm = lgbm_fit(lp, full_tr[F], full_tr.y, n_estimators=nl)
    xm = xgb_fit(xp, full_tr[F], full_tr.y, n_estimators=nx)
    lm.booster_.save_model(str(MODELS_DIR / "oulad_completion_lgbm.txt"))
    xm.save_model(str(MODELS_DIR / "oulad_completion_xgb.json"))
    log("final models refit and saved")

    # ---- 4. single evaluation on the held-out cohort
    pl = lm.predict_proba(test[F])[:, 1]
    px = xm.predict_proba(test[F])[:, 1]
    final = {"lightgbm": pl, "xgboost": px, "ensemble": (pl + px) / 2}[choice]
    base_lr = logreg(full_tr[F], full_tr.y).predict_proba(test[F])[:, 1]
    lgb_default = lgbm_fit({}, full_tr[F], full_tr.y, n_estimators=300).predict_proba(test[F])[:, 1]
    yt = test.y.to_numpy()
    res = {
        "majority_class": metrics(yt, np.full(len(yt), float(full_tr.y.mean() >= 0.5)), 0.5),
        "rule_baseline": metrics(yt, rule_baseline(test), 0.5),
        "logistic_regression": metrics(yt, base_lr, 0.5),
        "lightgbm_untuned": metrics(yt, lgb_default, 0.5),
        "lightgbm_tuned": metrics(yt, pl, 0.5),
        "xgboost_tuned": metrics(yt, px, 0.5),
        "ensemble_tuned": metrics(yt, (pl + px) / 2, 0.5),
        f"FINAL_{choice}@0.5": metrics(yt, final, 0.5),
        f"FINAL_{choice}@val_threshold": metrics(yt, final, thr),
    }
    out["test"] = res
    out["test_by_horizon"] = {
        int(h): metrics(yt[test.H.to_numpy() == h], final[test.H.to_numpy() == h], thr) for h in sorted(test.H.unique())
    }
    out["test_by_type"] = {
        k: metrics(yt[test.is_tma.to_numpy() == v], final[test.is_tma.to_numpy() == v], thr)
        for k, v in (("TMA", 1), ("CMA", 0))
    }
    log("test:", {k: round(v["accuracy"], 4) for k, v in res.items()})

    # ---- ablations (same tuned LightGBM settings, different feature sets) — reported, not used for selection
    abl = {}
    for name, fs in FEATURE_SETS.items():
        m = lgbm_fit(lp, full_tr[fs], full_tr.y, n_estimators=nl)
        abl[name] = metrics(yt, m.predict_proba(test[fs])[:, 1], 0.5)
    out["ablation_feature_sets_lightgbm"] = abl
    log("ablations:", {k: round(v["roc_auc"], 4) for k, v in abl.items()})

    # ---- explainability: mean |SHAP| on a test sample
    smp = test[F].sample(min(3000, len(test)), random_state=SEED)  # exact SHAP is ~20 ms/row on deep trees
    contrib = np.asarray(lm.predict(smp, pred_contrib=True))[:, :-1]
    out["shap_mean_abs_lightgbm"] = dict(
        sorted(zip(F, np.abs(contrib).mean(0).round(5).tolist(), strict=True), key=lambda x: -x[1])
    )
    out["gain_importance_lightgbm"] = dict(
        sorted(zip(F, lm.booster_.feature_importance("gain").round(1).tolist(), strict=True), key=lambda x: -x[1])
    )

    keep = ["id_student", "code_module", "code_presentation", "id_assessment", "H", "is_tma", "y"]
    pred = test[keep].copy()
    pred["p_lightgbm"], pred["p_xgboost"], pred["p_final"] = pl, px, final
    pred["p_logreg"], pred["p_rule"] = base_lr, rule_baseline(test)
    pred.to_csv(REPORTS_DIR / "oulad_test_predictions.csv.gz", index=False)
    with open(REPORTS_DIR / "oulad_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    card = {
        "model": choice,
        "threshold": thr,
        "features": F,
        "categorical": [c for c in F if c in CATEGORICAL],
        "lightgbm_file": "oulad_completion_lgbm.txt",
        "xgboost_file": "oulad_completion_xgb.json",
        "trained_on": TRAIN_PRESENTATIONS,
        "label": "1 = submitted on or before the deadline",
        "prediction_time": "end of day t = deadline - H, H in {1,3,7,14}",
    }
    with open(MODELS_DIR / "model_card.json", "w") as f:
        json.dump(card, f, indent=2)
    log("done")


if __name__ == "__main__":
    main()
