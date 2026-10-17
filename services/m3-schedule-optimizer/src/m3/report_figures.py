"""Figures for the OULAD completion-model report (reads reports/oulad_results.json + predictions)."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import REPORTS_DIR

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#c3c2b7"
plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
    }
)

NAMES = {
    "majority_class": "Always 'not on time'",
    "rule_baseline": "Simple rule (past on-time rate)",
    "logistic_regression": "Logistic regression",
    "lightgbm_untuned": "LightGBM (default settings)",
    "lightgbm_tuned": "LightGBM (tuned)",
    "xgboost_tuned": "XGBoost (tuned)",
    "ensemble_tuned": "LightGBM + XGBoost average",
}


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / name, dpi=160)
    plt.close(fig)


def model_comparison(r):
    res = r["test"]
    final_key = next(k for k in res if k.startswith("FINAL_") and k.endswith("@val_threshold"))
    rows = [(NAMES[k], v["accuracy"]) for k, v in res.items() if k in NAMES]
    rows.append((f"FINAL — {r['tuning']['chosen']} (tuned threshold)", res[final_key]["accuracy"]))
    rows.sort(key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    y = np.arange(len(rows))
    colors = [S1 if n.startswith("FINAL") else MUTED for n, _ in rows]
    ax.barh(y, [v * 100 for _, v in rows], color=colors, height=0.6)
    for i, (_, v) in enumerate(rows):
        ax.text(v * 100 + 0.6, i, f"{v * 100:.1f}%", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [n for n, _ in rows])
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Accuracy on held-out 2014J cohort (%)")
    ax.set_title("Model accuracy — task completed on time?", loc="left", fontsize=11, color=INK)
    _save(fig, "fig1_model_accuracy.png")


def by_horizon(r):
    h = r["test_by_horizon"]
    ks = sorted(h, key=int)
    acc = [h[k]["accuracy"] * 100 for k in ks]
    base = [max(h[k]["on_time_rate"], 1 - h[k]["on_time_rate"]) * 100 for k in ks]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    x = np.arange(len(ks))
    ax.bar(x, acc, width=0.55, color=S1, label="Final model")
    ax.scatter(x, base, marker="_", s=900, color=INK2, linewidths=2, label="Always guess the majority", zorder=3)
    for i, v in enumerate(acc):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_xticks(x, [f"{k} day{'s' if k != '1' else ''} before" for k in ks])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Accuracy by how far ahead the prediction is made", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    _save(fig, "fig2_accuracy_by_horizon.png")


def calibration(pred):
    p, y = pred["p_final"].to_numpy(), pred["y"].to_numpy()
    edges = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, edges) - 1, 0, 9)
    mp = [p[idx == b].mean() for b in range(10) if (idx == b).any()]
    my = [y[idx == b].mean() for b in range(10) if (idx == b).any()]
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.5, ls="--", label="Perfect calibration")
    ax.plot(mp, my, color=S1, lw=2, marker="o", ms=6, label="Final model")
    ax.set_xlabel("Predicted probability of on-time completion")
    ax.set_ylabel("Actual on-time rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_title("Are the probabilities trustworthy?", loc="left", fontsize=11)
    _save(fig, "fig3_calibration.png")


def importance(r, top=15):
    s = list(r["shap_mean_abs_lightgbm"].items())[:top][::-1]
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.barh(range(len(s)), [v for _, v in s], color=S1, height=0.6)
    ax.set_yticks(range(len(s)), [k for k, _ in s])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean |SHAP| (impact on predicted log-odds)")
    ax.set_title(f"Top {top} drivers of the prediction", loc="left", fontsize=11)
    _save(fig, "fig4_feature_importance.png")


def main(reports_dir=None):
    global REPORTS_DIR
    if reports_dir:
        REPORTS_DIR = Path(reports_dir)
    with open(REPORTS_DIR / "oulad_results.json") as f:
        r = json.load(f)
    pred = pd.read_csv(REPORTS_DIR / "oulad_test_predictions.csv.gz")
    model_comparison(r)
    by_horizon(r)
    calibration(pred)
    importance(r)
    print("figures written to", REPORTS_DIR)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
