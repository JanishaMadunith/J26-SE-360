# Module 3 — Task-completion model on OULAD (results, 17 Oct 2026)

**Question the model answers:** *Will this task be completed by its deadline?* — asked 1, 3, 7 or 14 days before the deadline, using only information available at that moment.

## Headline (held-out 2014J cohort — never used for training, tuning or model choice)

| Metric | Final model (XGBoost) |
|---|---|
| Accuracy | **83.4%** (majority-class baseline 52.7%) |
| ROC-AUC | **0.916** |
| Macro F1 | 0.834 |
| At-risk tasks caught (recall, "not on time") | 81.0% |
| Precision when flagging at-risk | 86.7% |
| Brier score / calibration error (ECE) | 0.114 / 0.021 — probabilities are well calibrated |

## All models on the same test set

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Always "not on time" | 52.7% | 0.500 |
| Simple rule (past on-time rate) | 66.8% | 0.766 |
| Logistic regression | 81.9% | 0.900 |
| LightGBM, default settings | 83.1% | 0.913 |
| LightGBM, tuned | 83.4% | 0.915 |
| **XGBoost, tuned (final)** | **83.4%** | **0.916** |
| LightGBM + XGBoost average | 83.5% | 0.915 |

The final model was chosen on the **2014B validation cohort** (XGBoost had the best validation AUC, 0.9340), not on the test results.

## By prediction horizon and task type

| Days before deadline | 1 | 3 | 7 | 14 |
|---|---|---|---|---|
| Accuracy | 83.9% | 83.8% | 83.3% | 82.9% |
| ROC-AUC | 0.918 | 0.916 | 0.911 | 0.904 |

| Task type | Accuracy | ROC-AUC |
|---|---|---|
| Tutor-marked assignments (TMA) | 79.3% | 0.859 |
| Computer-marked assessments (CMA) | 94.2% | 0.974 |

## What drives the prediction (mean |SHAP|, top 8)

1. Task type (TMA vs CMA)
2. Student's past on-time rate **for the same type of task**
3. Days since the student was last active
4. Task weight (importance)
5. Number of earlier tasks still outstanding
6. Prediction horizon
7. Overdue tasks across all the student's modules (workload)
8. How early/late the student submitted their last task

→ History, recent engagement and workload matter most — exactly the signals Module 3 will have in the app (task log + Module 2 wellness).

## Ablation (same tuned settings, test ROC-AUC)

| Feature set | AUC |
|---|---|
| Behaviour only (final — used in the app) | 0.915 |
| + demographics | 0.916 |
| + prior scores | 0.915 |
| All | 0.916 |

Demographics add only +0.001, so the final model deliberately leaves them out (fairer, and the app doesn't collect them).

## Method (for the report)

- **Data:** OULAD (Kuzilek et al., 2017) — 32,593 student registrations, 22 module presentations. Each TMA/CMA assessment is a task with a deadline.
- **Samples:** 663,420 (student, task, horizon) cases; label = submitted on or before the deadline. Tasks already submitted at prediction time are excluded; students who withdraw before the deadline count as not completed.
- **No leakage:** every feature uses data up to the prediction day only; an independent audit (`tests/research/test_no_leakage.py`) recomputes features from the raw files and passes all 11 checks.
- **Split by time:** train 2013B + 2013J (+ 2014B for the final fit), validate 2014B, test 2014J.
- **Features (65):** task (type, weight, timing, horizon), student history (on-time rates, outstanding/overdue work, earliness), engagement (clicks, active days, recency, trends, cohort-relative percentiles, activity types).
- **Tuning:** Optuna (TPE) on validation ROC-AUC — 26 LightGBM and 6 XGBoost trials; XGBoost final: depth 10, learning rate 0.012, 825 trees.
- **Reproduce:** `DATA_DIR=../../../07-datasets python -m m3.train_oulad` (from `services/m3-schedule-optimizer`, with `src` on PYTHONPATH).

## Limitations

- OULAD students are UK distance learners and tasks are course assessments — a **proxy** for personal tasks. The model validates the method; the final app model will be retrained/fine-tuned on pilot task logs (task-event log already recorded by the core API).
- No wellness inputs in OULAD — the link to Module 2's fatigue/focus is tested in simulation and the pilot.

## Files

`fig1_model_accuracy.png` · `fig2_accuracy_by_horizon.png` · `fig3_calibration.png` · `fig4_feature_importance.png` · `oulad_results.json` (all metrics) · `oulad_test_predictions.csv.gz` (gitignored) · trained models in `../models/` (gitignored; `model_card.json` committed)
