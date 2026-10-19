# Module 3 — Status for Progress Presentation 1 and plan to the final

**Module:** Predictive Adaptive Schedule Optimization · **Owner:** G.W.D.D.N. Kumarasinghe (IT23267336) ·
**Status date:** 19 Oct 2026

## 1. Datasets and how each one is used

| Dataset | What it contains | Used for | Status |
|---|---|---|---|
| **OULAD** (Kuzilek et al., 2017, CC BY 4.0) | 32,593 students, assessments with deadlines, submissions, 10M+ daily activity records | Training and testing the **completion model** (663,420 cases) | ✅ Used |
| **MS-LaTTE** (Jauhar et al., LREC 2022) | 10,101 real to-do titles, each judged by 5 people for *when* it is done | Training the **time-of-day model** | ✅ Used |
| **PMData** (Thambawita et al., 2020, CC BY 4.0) | 16 adults, 1,724 days of fatigue, mood, readiness, stress, sleep | **Wellness stand-in** for Module 2 (hourly energy / focus / fatigue in Module 2's format) | ✅ Used |
| **ATUS 2025** (US BLS) | 24-hour time diaries | **Default busy hours** (work / class / travel) for a working adult and a university student | ✅ Used |
| **StudentLife** (Wang et al., 2014) | 48 students over 10 weeks: deadlines, stress and sleep reports | **Evidence** that deadline load raises stress and cuts sleep, and effect sizes for the simulator | ✅ Used |

No public dataset links task outcomes with wellness, which supports the novelty claim and the three-layer data
strategy: public proxy data (now) → simulation (November) → own pilot data (after the app collects it).

## 2. Models and accuracy

### Completion model — "Will this task be finished on time?"
* **Data:** 663,420 cases (each assessment asked 14, 7, 3 and 1 day(s) before its deadline), 65 leakage-safe
  features; an 11-check leakage audit passes.
* **Split by semester:** train 2013B + 2013J, tune on 2014B, final test once on **2014J** (never seen).
* **Model:** XGBoost tuned with Optuna (29 trials logged), threshold chosen on validation.

| Model (same 2014J test set) | Accuracy | ROC-AUC |
|---|---|---|
| Always "not on time" | 52.7 % | 0.500 |
| Simple rule (past on-time rate) | 66.8 % | 0.766 |
| Logistic regression | 81.9 % | 0.900 |
| LightGBM tuned | 83.4 % | 0.915 |
| **XGBoost tuned (final)** | **83.4 %** | **0.916** |

At-risk recall 81.0 %, at-risk precision 86.7 %, calibration error (ECE) 0.021. Accuracy 83.9 / 83.8 / 83.3 / 82.9 %
at 1 / 3 / 7 / 14 days before the deadline.

**Cross-validation (leakage-safe):**
* Rolling over semesters (learn from the past, test on the next): **84.0 % ± 1.4** accuracy, AUC 0.922 ± 0.010.
* Student-grouped 5-fold (no student in both training and test): **86.1 % ± 0.1**, AUC 0.938.

### Time-of-day model — "When in the day is this task usually done?"
TF-IDF of the title and its list / category + logistic regression; settings chosen on a validation part of the
training data.
**56.3 %** on the held-out 20 % (4 classes; always guessing "evening" gives 41.6 %), **84.7 %** top-2;
5-fold CV 56.2 % ± 0.6. The task is ambiguous even for people: one annotator matches the majority of the other
annotators only **37.1 %** of the time, so the model agrees with the crowd better than a single person does. Examples: "morning run" → morning, "team meeting" → afternoon, "buy groceries" → evening,
"watch a movie" → night.

### Supporting results
* **StudentLife:** within the same student, each extra deadline in the next 3 days adds **+0.18** to stress (1–5)
  and removes **0.25 h** of sleep (r = 0.19 and −0.19). Average sleep: 6.3 h with no deadlines → 5.5 h with 3 or more.
* **ATUS:** default busy hours 07:30–17:15 (working adult) and 09:00–15:30 (university student), weekdays.

## 3. Proof in MLflow

`mlflow ui --backend-store-uri sqlite:///mlflow.db` → http://127.0.0.1:5000 (Model training mode):
* **M3 - Task completion prediction (OULAD):** final model run (parameters, metrics, charts, model file, evidence
  notebook), 7 baselines, 29 tuning trials, **model history** (11 loadable models, registry `m3-task-completion`
  v1–v11, v11 = `@champion`), **cross-validation** run with fold charts.
* **M3 - Task time-of-day (MS-LaTTE):** metrics, baseline and saved model.
* **M3 - Supporting datasets:** PMData, ATUS and StudentLife runs with their figures.

## 4. Demo

`PYTHONPATH=src:. uvicorn app.main:app --port 8103`, then open **http://localhost:8103/demo**:
1. **Will it be done on time?** Real unseen OULAD cases: prediction vs. what really happened, with the reasons
   (SHAP); "what if" sliders for a user's behaviour.
2. **Plan my days:** tasks → chance on time (completion model) → best time of day (time-of-day model) → wellness
   (PMData) and busy hours (ATUS) → schedule on a timeline with at-risk tasks marked.
3. **Models & data:** summary of everything above. The same endpoints are in **/docs** (Swagger).

## 5. Plan to the final evaluation

| When | Work | Output |
|---|---|---|
| Oct (PP1) | ✅ Datasets, completion model, time-of-day model, wellness stand-in, busy hours, scheduler with models, demo, MLflow | This status |
| Nov wk 1–2 | **OR-Tools CP-SAT optimiser** behind the same interface (deadlines, no overlaps, energy fit, minimal moves when re-planning) | `scheduling/cpsat.py` + tests |
| Nov wk 2–3 | **Simulator**: users built from ATUS days, MS-LaTTE tasks, PMData wellness and StudentLife stress effects | `m3/simulate.py` |
| Nov wk 3–4 | **Evaluation**: plain to-do list vs. greedy scheduler vs. CP-SAT — tasks finished on time, deadline misses, overload hours, re-plans | results chapter |
| Dec | Connect **Module 2's real forecast** (agree hourly slots with Samarathunga) and the core API task-event log → model features | integration |
| Jan → | **Pilot**: collect real task logs; fine-tune the completion model on them; final evaluation on target users | final report |

**Risks:** OULAD is a proxy (study assignments) → mitigated by behaviour-based features and pilot fine-tuning;
Module 2 not ready → PMData stand-in already works in Module 2's format; small pilot → simulator gives scale.
