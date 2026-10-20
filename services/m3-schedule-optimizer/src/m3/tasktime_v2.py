# ruff: noqa: N803, N806, RUF005, RUF046  (ML naming: X, C, S as in the papers)
"""Time-of-day model, round 2: can a language model beat TF-IDF on MS-LaTTE?

Same data, same 80/20 test split and the same validation part (20% of training) as m3.tasktime, so every number is
comparable with the current model (v1, 56.3%).

  v2  TF-IDF + LR trained on ALL annotator votes (soft labels: a task voted 3x evening, 2x afternoon becomes two
      weighted examples) instead of only the winning vote
  v3  GloVe word vectors (spaCy en_core_web_lg) of the title and list name + LR
  v4  v3 with all votes
  v5  RoBERTa-base sentence vectors (frozen, mean of the last layer; spaCy en_core_web_trf) + LR
  v6  v5 with all votes
  v7  RoBERTa-base fine-tuned end to end (all 125M weights) on all votes: 3 epochs, AdamW lr 3e-5, batch 32,
      warm-up 60 steps then linear decay; the epoch with the best validation accuracy is kept
  v8  ensemble: weighted average of the probabilities of v1, v2, v4 and v7; weights (0-3 each) chosen on validation

Extra training dependencies (not needed by the service):
    pip install torch spacy spacy-curated-transformers
    pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl
    pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl

Usage (from services/m3-schedule-optimizer):
    PYTHONPATH=src python -m m3.tasktime_v2 --out reports/tasktime_v2      # ~35 min on 2 CPU cores (fine-tune ~27 min)
    PYTHONPATH=src python -m m3.log_tasktime_v2                            # log the results to MLflow
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

from .config import REPORTS_DIR, SEED
from .tasktime import PARTS, dataset, evaluate, human_baseline, make_model

K = len(PARTS)


def splits(y):
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)
    fi, va = train_test_split(tr, test_size=0.2, random_state=SEED, stratify=y[tr])
    return tr, te, fi, va


def nice(text: str) -> str:
    title, lst = text.split(" || list: ")
    return f"{title} (list: {lst})"


def embeddings(texts):
    """GloVe (title vector + list vector) and frozen RoBERTa (mean of the last layer)."""
    import spacy

    lg = spacy.load("en_core_web_lg", exclude=["tagger", "parser", "attribute_ruler", "lemmatizer", "ner", "senter"])
    pairs = [t.split(" || list: ") for t in texts]
    glove = np.hstack([np.stack([lg(a).vector for a, _ in pairs]), np.stack([lg(b).vector for _, b in pairs])])
    trf = spacy.load("en_core_web_trf", exclude=["tagger", "parser", "attribute_ruler", "lemmatizer", "ner"])
    rob = np.stack(
        [
            np.asarray(d._.trf_data.last_hidden_layer_state.dataXd).mean(0)
            for d in trf.pipe(map(nice, texts), batch_size=64)
        ]
    )
    return glove, rob


def linear_runs(T, G, R, y, S, tr, te, fi, va):
    """v1-v6: C chosen by validation accuracy (ties: log-loss), refit on the training part, scored on test."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def soft(i):  # one weighted row per part of day
        rep, lab, w = np.repeat(i, K), np.tile(np.arange(K), len(i)), S[i].ravel()
        return rep[w > 0], lab[w > 0], w[w > 0]

    def fit(build, X, i, use_soft):
        m = build()
        if use_soft:
            r, lab, w = soft(i)
            return m.fit(X[r], lab, **{m.steps[-1][0] + "__sample_weight": w})
        return m.fit(X[i], y[i])

    dense = lambda C: lambda: make_pipeline(StandardScaler(), LogisticRegression(C=C / 50, max_iter=3000))  # noqa: E731
    tfidf = lambda C: lambda: make_model(C)  # noqa: E731
    specs = {
        "v1": (tfidf, T, False),
        "v2": (tfidf, T, True),
        "v3": (dense, G, False),
        "v4": (dense, G, True),
        "v5": (dense, R, False),
        "v6": (dense, R, True),
    }
    out = {}
    for name, (build, X, use_soft) in specs.items():
        best = None
        for C in (0.25, 0.5, 1, 2, 4):  # noqa: N806
            m = fit(build(C), X, fi, use_soft)
            pv = m.predict_proba(X[va])
            key = ((pv.argmax(1) == y[va]).mean(), -log_loss(y[va], pv, labels=range(K)))
            if best is None or key > best[0]:
                best = (key, C, pv)
        _, C, pv = best  # noqa: N806
        pt = fit(build(C), X, tr, use_soft).predict_proba(X[te])
        out[name] = {"C": C, "pv": pv, "pt": pt}
        print(f"{name} C={C} test acc {(pt.argmax(1) == y[te]).mean():.4f}", flush=True)
    return out


def finetune(texts, y, S, te, fi, va, epochs=3, lr=3e-5, bs=32):
    """v7: fine-tune RoBERTa-base (weights from spaCy en_core_web_trf) with cross-entropy on the vote shares."""
    import spacy
    import torch
    from curated_transformers.models.attention import AttentionMask

    torch.manual_seed(SEED)
    nlp = spacy.load("en_core_web_trf", exclude=["tagger", "parser", "attribute_ruler", "lemmatizer", "ner"])
    nodes = {x.name: x for x in nlp.get_pipe("transformer").model.walk()}
    proc = nodes["byte_bpe_encoder"].attrs["byte_bpe_processor"]
    enc = nodes["pytorch"].shims[0]._model
    bos, eos, pad = proc.piece_id("<s>"), proc.piece_id("</s>"), 1
    ids = [[bos] + proc.encode_as_ids(" " + nice(t))[:48] + [eos] for t in texts]

    def batch(ii):
        L = max(len(ids[i]) for i in ii)  # noqa: N806
        X = torch.full((len(ii), L), pad)  # noqa: N806
        M = torch.zeros((len(ii), L), dtype=torch.bool)  # noqa: N806
        for r, i in enumerate(ii):
            X[r, : len(ids[i])] = torch.tensor(ids[i])
            M[r, : len(ids[i])] = True
        return X, M

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc, self.drop, self.head = enc, torch.nn.Dropout(0.1), torch.nn.Linear(768, K)

        def forward(self, X, M):  # noqa: N803
            h = self.enc(X, attention_mask=AttentionMask(M)).last_hidden_layer_states
            m = M.unsqueeze(-1).float()
            return self.head(self.drop((h * m).sum(1) / m.sum(1)))

    net = Net()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * int(np.ceil(len(fi) / bs))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1, (k + 1) / 60) * max(0, 1 - k / steps))

    def probs(ii):
        net.eval()
        with torch.no_grad():
            out = [torch.softmax(net(*batch(ii[b : b + 128])), -1).numpy() for b in range(0, len(ii), 128)]
        net.train()
        return np.vstack(out)

    rng, best, t0, curve = np.random.default_rng(SEED), (-1.0, None, None), time.time(), []
    for ep in range(epochs):
        order = rng.permutation(fi)
        for b in range(0, len(order), bs):
            ii = order[b : b + bs]
            target = torch.tensor(S[ii], dtype=torch.float32)
            loss = -(target * torch.log_softmax(net(*batch(ii)), -1)).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
        pv = probs(va)
        acc = float((pv.argmax(1) == y[va]).mean())
        curve.append(acc)
        print(f"v7 epoch {ep + 1} val acc {acc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if acc > best[0]:
            best = (acc, pv, probs(te))
    return {"pv": best[1], "pt": best[2], "val_curve": curve, "epochs": epochs, "lr": lr, "batch": bs}


def ensemble(runs, y, va, use=("v1", "v2", "v4", "v7")):
    best = (0.0, None)
    for w in itertools.product((0, 1, 2, 3), repeat=len(use)):
        if sum(w):
            acc = (sum(wi * runs[n]["pv"] for wi, n in zip(w, use, strict=True)).argmax(1) == y[va]).mean()
            if acc > best[0]:
                best = (acc, w)
    w = dict(zip(use, best[1], strict=True))
    pt = sum(w[n] * runs[n]["pt"] for n in use) / sum(w.values())
    pv = sum(w[n] * runs[n]["pv"] for n in use) / sum(w.values())
    return {"weights": w, "pv": pv, "pt": pt}


NAMES = {
    "v1": ("TF-IDF + logistic regression (current)", "word and letter counts", "majority vote"),
    "v2": ("TF-IDF + LR, trained on all votes", "word and letter counts", "all votes (soft labels)"),
    "v3": ("GloVe word vectors + LR", "pre-trained word meanings (spaCy en_core_web_lg)", "majority vote"),
    "v4": ("GloVe + LR, all votes", "pre-trained word meanings", "all votes"),
    "v5": ("RoBERTa (frozen) + LR", "RoBERTa-base sentence vectors, not trained", "majority vote"),
    "v6": ("RoBERTa (frozen) + LR, all votes", "RoBERTa-base sentence vectors", "all votes"),
    "v7": ("RoBERTa fine-tuned", "RoBERTa-base, all 125M weights trained", "all votes"),
    "v8": (
        "Ensemble (TF-IDF + GloVe + fine-tuned RoBERTa)",
        "weighted average of v1, v2, v4, v7",
        "weights chosen on validation",
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPORTS_DIR / "tasktime_v2"))
    ap.add_argument("--epochs", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    texts, y, S = dataset()  # noqa: N806
    tr, te, fi, va = splits(y)
    G, R = embeddings(texts)  # noqa: N806
    runs = linear_runs(np.array(texts, dtype=object), G, R, y, S, tr, te, fi, va)
    runs["v7"] = finetune(texts, y, S, te, fi, va, epochs=a.epochs)
    runs["v8"] = ensemble(runs, y, va)
    major = np.bincount(y[tr]).argmax()
    base = np.zeros((len(te), K))
    base[:, major] = 1
    models = []
    for k, (name, feat, lab) in NAMES.items():
        r = runs[k]
        extra = {x: r[x] for x in ("C", "weights", "val_curve", "epochs", "lr", "batch") if x in r}
        models.append(
            {
                "key": k,
                "name": f"{k} {name}",
                "features": feat,
                "labels": lab,
                "val_accuracy": round(float((r["pv"].argmax(1) == y[va]).mean()), 4),
                **extra,
                **{m: round(v, 4) for m, v in evaluate(y[te], r["pt"], S[te]).items()},
            }
        )
    summary = {
        "dataset": "MS-LaTTE",
        "split": f"same 80/20 split as m3.tasktime (seed {SEED}); validation = 20% of the training part",
        "n_test": int(len(te)),
        "baselines": {
            "always_most_common": round(evaluate(y[te], base, S[te])["accuracy"], 4),
            "one_person_vs_others": round(human_baseline(), 4),
        },
        "models": models,
    }
    (out / "results.json").write_text(json.dumps(summary, indent=1))
    for m in models:
        print(f"{m['name']:60s} acc {m['accuracy']:.4f}  top-2 {m['top2_accuracy']:.4f}")


if __name__ == "__main__":
    main()
