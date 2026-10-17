"""Independent leakage audit for the OULAD completion dataset.

Recomputes a sample of features directly from the raw CSVs (not via features_oulad's
arrays) and checks every rule that keeps the evaluation honest.

Run (needs requirements-train.txt + the OULAD data):
    DATA_DIR=... python tests/research/test_no_leakage.py [samples.pkl] [reports_dir]
Not collected by the default pytest run (serving tests only).
"""
# ruff: noqa: N803, N806  -- ML convention: X (features), F (feature list), H (horizon) are uppercase

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from m3.config import DATA_DIR, MODELS_DIR, REPORTS_DIR, TEST_PRESENTATION, TRAIN_PRESENTATIONS  # noqa: E402
from m3.data_oulad import load_oulad  # noqa: E402
from m3.features_oulad import FEATURE_SETS, build_dataset  # noqa: E402

FORBIDDEN = {"final_result", "score", "date_submitted", "date_unregistration", "y", "id_student", "id_assessment"}


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    return bool(ok)


def main(samples=None, reports=None):
    reports = Path(reports) if reports else REPORTS_DIR
    t = load_oulad(DATA_DIR)
    df = pd.read_pickle(samples) if samples else build_dataset(t, verbose=False)
    F = FEATURE_SETS["behaviour"]
    ok = True

    ok &= check("no forbidden columns in the model's features", not (FORBIDDEN & set(F)), str(FORBIDDEN & set(F)))
    ok &= check(
        "train and test cohorts are disjoint",
        not set(df[df.code_presentation.isin(TRAIN_PRESENTATIONS)].code_presentation) & {TEST_PRESENTATION},
    )
    ok &= check(
        "no sample's task was already submitted at prediction time",
        ((df.date_submitted.isna()) | (df.date_submitted > df.t)).all(),
    )
    ok &= check(
        "every sample's student is registered at prediction time",
        (
            (df.date_registration.isna() | (df.date_registration <= df.t))
            & (df.date_unregistration.isna() | (df.date_unregistration > df.t))
        ).all(),
    )
    lab = (df.date_submitted.notna() & (df.date_submitted <= df.deadline)).astype(int)
    ok &= check("label = submitted on or before deadline", (lab == df.y).all())

    # Recompute click features from the raw VLE table for a random sample of test rows
    v = t["studentVle"]
    smp = df[df.code_presentation == TEST_PRESENTATION].sample(300, random_state=7)
    bad = 0
    for _, r in smp.iterrows():
        x = v[
            (v.code_module == r.code_module)
            & (v.code_presentation == r.code_presentation)
            & (v.id_student == r.id_student)
        ]
        tot = x.loc[x.date <= r.t, "sum_click"].sum()
        last7 = x.loc[(x.date <= r.t) & (x.date > r.t - 7), "sum_click"].sum()
        future = x.loc[(x.date > r.t) & (x.date <= r.deadline), "sum_click"].sum()
        if abs(tot - r.clicks_total) > 0.5 or abs(last7 - r.clicks_7) > 0.5:
            bad += 1
        if future > 0 and abs((tot + future) - r.clicks_total) < 0.5 and abs(future) > 0.5:
            bad += 1  # feature would equal the leaked (post-cutoff) total
    ok &= check("click features match raw data up to day t only (300 random test rows)", bad == 0, f"mismatches={bad}")

    # Recompute history feature for a sample
    sa = t["studentAssessment"].merge(t["assessments"], on="id_assessment")
    bad = 0
    for _, r in smp.head(150).iterrows():
        h = sa[
            (sa.id_student == r.id_student)
            & (sa.code_module == r.code_module)
            & (sa.code_presentation == r.code_presentation)
            & (sa.assessment_type != "Exam")
            & (sa.date <= r.t)
            & (sa.id_assessment != r.id_assessment)
            & (sa.is_banked == 0)
        ]
        on_time = int((h.date_submitted <= h.date).sum())
        if on_time != int(r.n_prior_on_time):
            bad += 1
    ok &= check("prior on-time counts match raw data (150 random test rows)", bad == 0, f"mismatches={bad}")

    # Recompute headline metrics from the saved predictions
    pf = reports / "oulad_test_predictions.csv.gz"
    if pf.exists():
        pred = pd.read_csv(pf)
        card = {}
        if (MODELS_DIR / "model_card.json").exists():
            with open(MODELS_DIR / "model_card.json") as f:
                card = json.load(f)
        thr = card.get("threshold", 0.5)
        with open(reports / "oulad_results.json") as f:
            res = json.load(f)
        key = next(k for k in res["test"] if k.endswith("@val_threshold"))
        acc = accuracy_score(pred.y, pred.p_final >= thr)
        auc = roc_auc_score(pred.y, pred.p_final)
        ok &= check(
            "saved test predictions reproduce reported accuracy",
            abs(acc - res["test"][key]["accuracy"]) < 1e-9,
            f"acc={acc:.4f}",
        )
        ok &= check(
            "saved test predictions reproduce reported AUC",
            abs(auc - res["test"][key]["roc_auc"]) < 1e-9,
            f"auc={auc:.4f}",
        )
        ok &= check(
            "prediction file covers exactly the test cohort",
            set(pred.code_presentation) == {TEST_PRESENTATION}
            and len(pred) == int((df.code_presentation == TEST_PRESENTATION).sum()),
        )
        rng = np.random.default_rng(0)
        ok &= check(
            "shuffled labels give chance-level AUC (sanity)",
            abs(roc_auc_score(rng.permutation(pred.y), pred.p_final) - 0.5) < 0.01,
        )
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(*sys.argv[1:3]) else 1)
