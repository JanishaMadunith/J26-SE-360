"""Task time-of-day model, trained on MS-LaTTE (Jauhar et al., LREC 2022; 10,101 real to-do titles).

Question: "When in the day do people usually do this task?"  ->  a preference over four parts of the day
(morning, afternoon, evening, night). The scheduler uses it as a soft preference when it picks a time slot.

Labels: each task was judged by 5 annotators, who could tick several of WD/WE x morning/afternoon/evening/night/
anytime. Votes are pooled into one share per part of day ("anytime" is spread evenly). The training target is the
part of day with most votes; the model outputs a probability for each part.

Model: TF-IDF of words (1-2 grams) and characters (2-5 grams) of the title + the list it is in (MS-LaTTE list name;
in the app, the task category mapped by CATEGORY_LIST) -> multinomial logistic regression.
Settings (C, class weighting) are chosen by log-loss on a validation part of the TRAINING data only.
Evaluation: held-out 20% test split + 5-fold cross-validation; compared with always predicting the most common part
and with people: how often one annotator agrees with the majority of the others (the task is ambiguous by nature).

Usage (from services/m3-schedule-optimizer):
    PYTHONPATH=src python -m m3.tasktime            # train, evaluate, save models/tasktime_model.joblib, log MLflow
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import DATA_DIR, MODELS_DIR, MODULE_ROOT, REPORTS_DIR, SEED

PARTS = ["morning", "afternoon", "evening", "night"]
# Local clock hours for each part of day (used by the scheduler).
PART_HOURS = {
    "morning": range(5, 12),
    "afternoon": range(12, 17),
    "evening": range(17, 21),
    "night": (21, 22, 23, 0, 1, 2, 3, 4),
}
MODEL_FILE = "tasktime_model.joblib"


def part_of_hour(hour: int) -> str:
    for p, hours in PART_HOURS.items():
        if hour in hours:
            return p
    return "night"


def load_mslatte(data_dir: Path = DATA_DIR) -> list[dict]:
    base = Path(data_dir) / "mslatte"
    for cand in (base / "raw" / "MS-LaTTE.json", base / "MS-LaTTE.json"):
        if cand.exists():
            return json.loads(cand.read_text())
    raise FileNotFoundError(f"MS-LaTTE.json not found under {base}/raw")


def votes(task: dict) -> np.ndarray:
    """Pooled annotator votes over PARTS (weekday and weekend together; 'anytime' spread evenly)."""
    v = np.zeros(len(PARTS))
    for j in task.get("TimeJudgements", []):
        if j.get("Known") != "yes":
            continue
        for t in filter(None, j.get("Times", "").split(",")):
            part = t.split("-", 1)[-1]
            if part == "anytime":
                v += 1 / len(PARTS)
            elif part in PARTS:
                v[PARTS.index(part)] += 1
    return v


# App task category -> the closest MS-LaTTE list name (the list a task sits in, e.g. "work", "school").
CATEGORY_LIST = {
    "study": "school",
    "work": "work",
    "exercise": "workout",
    "chores": "chores",
    "personal": "family",
    "social": "family",
    "other": "default list",
}


def model_text(title: str, list_name: str | None = None) -> str:
    """Model input: the task title plus the list / category it belongs to (context helps: 'call' in 'work' vs 'family')."""
    return f"{title.strip().lower()} || list: {(list_name or 'default list').strip().lower()}"


def annotator_parts(task: dict) -> list[int]:
    """Each annotator's preferred part of day (for the human-agreement baseline)."""
    out = []
    for j in task.get("TimeJudgements", []):
        if j.get("Known") != "yes":
            continue
        v = votes({"TimeJudgements": [j]})
        if v.sum():
            out.append(v)
    return out


def human_baseline(data_dir: Path = DATA_DIR, seed: int = SEED) -> float:
    """How often ONE annotator's choice matches the majority of the OTHER annotators (a human 'accuracy')."""
    rng = np.random.default_rng(seed)
    hits = []
    for t in load_mslatte(data_dir):
        ann = annotator_parts(t)
        if len(ann) < 2:
            continue
        k = int(rng.integers(len(ann)))
        rest = sum(a for i, a in enumerate(ann) if i != k)
        hits.append(int(np.argmax(ann[k])) == int(np.argmax(rest)))
    return float(np.mean(hits))


def dataset(data_dir: Path = DATA_DIR):
    texts, y, shares = [], [], []
    for t in load_mslatte(data_dir):
        v = votes(t)
        title = (t.get("TaskTitle") or "").strip().lower()
        if v.sum() == 0 or not title:
            continue
        texts.append(model_text(title, t.get("ListTitle")))
        y.append(int(np.argmax(v)))
        shares.append(v / v.sum())
    return texts, np.array(y), np.array(shares)


def make_model(C: float = 1.0, class_weight=None):  # noqa: N803
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline, make_union

    feats = make_union(
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
    )
    return make_pipeline(feats, LogisticRegression(C=C, max_iter=3000, class_weight=class_weight))


def evaluate(y, proba, shares) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    pred = proba.argmax(1)
    top2 = np.argsort(-proba, 1)[:, :2]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "top2_accuracy": float(np.mean([y[i] in top2[i] for i in range(len(y))])),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        # share of annotators who also chose the predicted part (how often people agree with the suggestion)
        "annotator_agreement": float(np.mean(shares[np.arange(len(y)), pred])),
    }


@lru_cache(maxsize=1)
def _load(path: str):
    import joblib

    return joblib.load(path)


def predict_parts(
    titles: list[str], models_dir: Path = MODELS_DIR, categories: list[str] | None = None
) -> list[dict[str, float]] | None:
    """Preference over PARTS for each title (and optional app category), or None if the model file is missing."""
    path = Path(models_dir) / MODEL_FILE
    if not path.exists():
        return None
    model = _load(str(path))
    cats = categories or [None] * len(titles)
    proba = model.predict_proba([model_text(t, CATEGORY_LIST.get(c, c)) for t, c in zip(titles, cats, strict=True)])
    return [{p: round(float(x), 4) for p, x in zip(PARTS, row, strict=True)} for row in proba]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracking-uri", default=f"sqlite:///{MODULE_ROOT / 'mlflow.db'}")
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args()
    import joblib
    from sklearn.model_selection import StratifiedKFold, train_test_split

    titles, y, shares = dataset()
    counts = Counter(y.tolist())
    print(
        f"{len(titles):,} tasks; most-voted part of day: "
        + ", ".join(f"{PARTS[k]} {v}" for k, v in sorted(counts.items()))
    )
    idx = np.arange(len(titles))
    tr, te = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)
    T = np.array(titles, dtype=object)  # noqa: N806
    from sklearn.metrics import log_loss

    fit_i, val_i = train_test_split(tr, test_size=0.2, random_state=SEED, stratify=y[tr])
    grid = {}
    for C in (0.5, 1.0, 2.0, 4.0):  # noqa: N806
        for cw in (None, "balanced"):
            mm = make_model(C, cw).fit(T[fit_i], y[fit_i])
            grid[(C, cw)] = log_loss(y[val_i], mm.predict_proba(T[val_i]), labels=range(len(PARTS)))
    best_c, best_cw = min(grid, key=grid.get)
    print(f"chosen on validation: C={best_c}, class_weight={best_cw}")
    m = make_model(best_c, best_cw).fit(T[tr], y[tr])
    res = evaluate(y[te], m.predict_proba(T[te]), shares[te])
    major = Counter(y[tr].tolist()).most_common(1)[0][0]
    base = np.zeros((len(te), len(PARTS)))
    base[:, major] = 1
    res_base = evaluate(y[te], base, shares[te])
    cv = []
    for f_tr, f_te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(idx, y):
        mm = make_model(best_c, best_cw).fit(T[f_tr], y[f_tr])
        cv.append(evaluate(y[f_te], mm.predict_proba(T[f_te]), shares[f_te]))
    cv_acc = [c["accuracy"] for c in cv]
    print(
        f"test accuracy {res['accuracy']:.3f} (top-2 {res['top2_accuracy']:.3f}) vs always-'{PARTS[major]}' "
        f"{res_base['accuracy']:.3f}; 5-fold CV {np.mean(cv_acc):.3f} ± {np.std(cv_acc, ddof=1):.3f}"
    )

    final = make_model(best_c, best_cw).fit(T, y)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, MODELS_DIR / MODEL_FILE)
    examples = [
        "go to the gym",
        "read research papers",
        "buy groceries",
        "team meeting",
        "call mom",
        "finish assignment",
        "pay electricity bill",
        "watch a movie",
        "morning run",
        "clean the kitchen",
    ]
    ex_cats = ["exercise", "study", "chores", "work", "personal", "study", "chores", "social", "exercise", "chores"]
    ex_in = [model_text(t, CATEGORY_LIST[c]) for t, c in zip(examples, ex_cats, strict=True)]
    ex = dict(zip(examples, final.predict_proba(ex_in).round(3).tolist(), strict=True))
    human = human_baseline()
    print(f"people: one annotator matches the majority of the others {human:.3f} of the time")
    summary = {
        "dataset": "MS-LaTTE (Jauhar et al., LREC 2022)",
        "n_tasks": len(titles),
        "chosen": {"C": best_c, "class_weight": str(best_cw)},
        "classes": PARTS,
        "class_counts": {PARTS[k]: v for k, v in sorted(counts.items())},
        "test": res,
        "baseline_most_common": {"part": PARTS[major], **res_base},
        "human_agreement_with_majority": human,
        "cv5_accuracy_mean": float(np.mean(cv_acc)),
        "cv5_accuracy_std": float(np.std(cv_acc, ddof=1)),
        "cv5": cv,
        "examples": ex,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "tasktime_results.json").write_text(json.dumps(summary, indent=2))
    for t, p in ex.items():
        print(f"  {t:<24} -> {PARTS[int(np.argmax(p))]:<9} {dict(zip(PARTS, p, strict=True))}")
    if a.no_mlflow:
        return
    import mlflow

    mlflow.set_tracking_uri(a.tracking_uri)
    mlflow.set_experiment("M3 - Task time-of-day (MS-LaTTE)")
    with mlflow.start_run(run_name="tasktime-tfidf-logreg"):
        mlflow.set_tags(
            {
                "mlflow.note.content": (
                    "Second Module 3 model: suggests the part of the day (morning / afternoon / evening / night) when "
                    "people usually do a task, from its title and list/category. Trained on 10,101 real to-do titles "
                    "from MS-LaTTE, each judged by 5 annotators. People themselves disagree a lot: one annotator matches "
                    "the others' majority only ~37% of the time, so the model (~56%) is better than a single person. "
                    "Used by the scheduler as a soft time-slot preference."
                )
            }
        )
        mlflow.log_params(
            {
                "dataset": "MS-LaTTE",
                "n_tasks": len(titles),
                "features": "tf-idf words 1-2 + chars 2-5 of title + list/category",
                "model": "LogisticRegression",
                "C": best_c,
                "class_weight": str(best_cw),
                "split": "80/20 stratified + 5-fold CV",
            }
        )
        mlflow.log_metrics({f"test_{k}": v for k, v in res.items()})
        mlflow.log_metrics({f"baseline_{k}": v for k, v in res_base.items()})
        mlflow.log_metric("human_agreement_with_majority", human)
        mlflow.log_metrics(
            {"cv5_accuracy_mean": float(np.mean(cv_acc)), "cv5_accuracy_std": float(np.std(cv_acc, ddof=1))}
        )
        mlflow.log_dict(summary, "tasktime_results.json")
        mlflow.log_artifact(str(MODELS_DIR / MODEL_FILE), "model")


if __name__ == "__main__":
    main()
