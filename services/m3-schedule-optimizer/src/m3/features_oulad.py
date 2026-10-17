"""Task-completion dataset built from OULAD.

Framing (mirrors what Module 3 does in the app):
  * task      = a TMA/CMA assessment with a deadline
  * question  = "Will this task be completed on time?"
  * asked at  = cutoff day t = deadline - H   (H in HORIZONS days before the deadline)

Leakage rules — every feature uses only information available at the end of day t:
  * VLE clicks with date <= t only.
  * Samples whose task was already submitted on or before t are dropped (nothing to predict).
  * A sample is included only if the student is registered at t (registered on/before t and not
    unregistered on/before t). Students who withdraw between t and the deadline are kept with
    label 0 — withdrawing is a real way of not completing the task.
  * History uses other assessments whose deadline <= t (their on-time status is then known) and
    submission dates <= t only.
  * `final_result` is never used. Scores are excluded from the main feature set (marking dates are
    unknown); they appear only in an optional ablation with an explicit 14-day release assumption.

Label: y = 1 if submitted on or before the deadline, else 0 (late, never submitted, or withdrew).
"""
# ruff: noqa: N803, N806  -- ML convention: X (features), F (feature list), H (horizon) are uppercase

import numpy as np
import pandas as pd

from .config import HORIZONS
from .data_oulad import KEYS

MP = ["code_module", "code_presentation"]
DMIN, DMAX = -30, 270
ND = DMAX - DMIN + 1
TOP_TYPES = ["homepage", "oucontent", "forumng", "subpage", "resource", "url", "quiz"]

TASK_FEATURES = [
    "H",
    "is_tma",
    "weight",
    "deadline",
    "deadline_frac",
    "a_order",
    "n_assess",
    "gap_prev",
    "gap_next",
    "module_length",
    "season_j",
    "code_module",
]
# task_hist_on_time / task_hist_n (earlier-cohort on-time rate of the same task slot) are computed
# but not used: tested on validation 17 Oct 2026, AUC 0.9311 vs 0.9314 without — no gain.
STUDENT_FEATURES = ["num_of_prev_attempts", "studied_credits", "date_registration", "n_concurrent_modules"]
HISTORY_FEATURES = [
    "n_prior",
    "n_prior_on_time",
    "prior_on_time_rate",
    "n_prior_late_known",
    "n_prior_outstanding",
    "prior_mean_days_early",
    "last_prior_on_time",
    "n_future_done",
    "n_upcoming_14_all",
    "n_upcoming_7_all",
    "n_overdue_all",
]
ENGAGEMENT_FEATURES = (
    [
        "clicks_total",
        "active_days_total",
        "clicks_pre_start",
        "clicks_3",
        "clicks_7",
        "clicks_14",
        "clicks_30",
        "active_7",
        "active_14",
        "days_since_last",
        "trend_7",
        "clicks_since_prev_deadline",
        "active_since_prev_deadline",
        "rel_clicks_7",
        "pct_clicks_7",
        "pct_clicks_total",
        "pct_clicks_14",
    ]
    + [f"{a}_14" for a in TOP_TYPES]
    + [f"{a}_total" for a in TOP_TYPES]
)
# v2 features — kept because together they improved validation AUC (tested on 2014B only)
EXTRA_V2 = [
    "prior_same_type_on_time_rate",
    "n_prior_same_type",
    "last_prior_days_early",
    "self_rel_clicks_7",
    "site_visits_14",
    "site_visits_7",
    "days_since_registration",
]
DEMOGRAPHIC_FEATURES = ["gender", "region", "highest_education", "imd_band", "age_band", "disability"]
SCORE_FEATURES = ["prior_mean_score_14d", "n_prior_scored_14d"]
CATEGORICAL = ["code_module", *DEMOGRAPHIC_FEATURES]

FEATURE_SETS = {
    # EXTRA_V2 added 17 Oct 2026: validation AUC 0.9316 -> 0.9325, accuracy 85.18% -> 85.48% (default LightGBM)
    "behaviour": TASK_FEATURES + STUDENT_FEATURES + HISTORY_FEATURES + ENGAGEMENT_FEATURES + EXTRA_V2,
}
FEATURE_SETS["behaviour+demographics"] = FEATURE_SETS["behaviour"] + DEMOGRAPHIC_FEATURES
FEATURE_SETS["behaviour+scores"] = FEATURE_SETS["behaviour"] + SCORE_FEATURES
FEATURE_SETS["all"] = FEATURE_SETS["behaviour"] + DEMOGRAPHIC_FEATURES + SCORE_FEATURES


def _enrolments(t):
    reg = t["studentInfo"].merge(t["studentRegistration"], on=KEYS, how="left")
    reg = reg.merge(t["courses"], on=MP, how="left")
    reg = reg.rename(columns={"module_presentation_length": "module_length"})
    reg["n_concurrent_modules"] = reg.groupby(["code_presentation", "id_student"])["id_student"].transform("size")
    reg = reg.reset_index(drop=True)
    reg["eid"] = np.arange(len(reg), dtype=np.int32)
    return reg


def _click_arrays(t, reg):
    """Dense per-enrolment daily click matrices and their cumulative sums."""
    v = t["studentVle"]
    key = reg[[*KEYS, "eid"]].copy()
    for c in MP:
        key[c] = key[c].astype(v[c].dtype)
    key["id_student"] = key["id_student"].astype("int32")
    v = v.merge(key, on=KEYS, how="inner")
    day = (v["date"].astype(np.int32) - DMIN).clip(0, ND - 1).to_numpy()
    flat = v["eid"].to_numpy().astype(np.int64) * ND + day
    n = len(reg)
    clicks = v["sum_click"].to_numpy().astype(np.float64)
    tot = np.bincount(flat, weights=clicks, minlength=n * ND).reshape(n, ND)
    site_type = t["vle"].set_index("id_site")["activity_type"]
    atype = v["id_site"].map(site_type).to_numpy()
    # each studentVle row is one (student, site, day) — counting rows gives distinct pages visited per day
    arrs = {"tot": tot, "sites": np.bincount(flat, minlength=n * ND).reshape(n, ND).astype(np.float64)}
    for a in TOP_TYPES:
        m = atype == a
        arrs[a] = np.bincount(flat[m], weights=clicks[m], minlength=n * ND).reshape(n, ND)
    out = {k: np.cumsum(x, axis=1).astype(np.float32) for k, x in arrs.items()}
    active = tot > 0
    out["active"] = np.cumsum(active, axis=1).astype(np.int16)
    la = np.where(active, np.arange(ND)[None, :], -1)
    out["last_active"] = np.maximum.accumulate(la, axis=1).astype(np.int16)
    return out


def _wsum(cum, eidx, tidx, w):
    """Sum over days (t-w, t] using cumulative arrays; tidx may be -1 (no data yet)."""
    w = np.asarray(w)
    hi = np.where(tidx >= 0, cum[eidx, np.clip(tidx, 0, None)], 0)
    lo_idx = tidx - w
    lo = np.where(lo_idx >= 0, cum[eidx, np.clip(lo_idx, 0, None)], 0)
    return (hi - lo).astype(np.float32)


def _task_table(t):
    a = t["assessments"]
    a = a[(a["assessment_type"] != "Exam") & a["date"].notna()].copy()
    a = a.sort_values([*MP, "date", "id_assessment"])
    a["a_order"] = a.groupby(MP).cumcount() + 1
    a["n_assess"] = a.groupby(MP)["id_assessment"].transform("size")
    a["gap_prev"] = a.groupby(MP)["date"].diff().fillna(a["date"])
    a["gap_next"] = -a.groupby(MP)["date"].diff(-1)
    a["is_tma"] = (a["assessment_type"] == "TMA").astype(np.int8)
    return a.rename(columns={"date": "deadline"})


def build_dataset(t, horizons=HORIZONS, verbose=True):
    reg = _enrolments(t)
    arr = _click_arrays(t, reg)
    a = _task_table(t)

    pairs = reg.merge(a, on=MP, how="inner")
    sub = t["studentAssessment"][["id_assessment", "id_student", "date_submitted", "is_banked", "score"]]
    pairs = pairs.merge(sub, on=["id_assessment", "id_student"], how="left")
    pairs = pairs[pairs["is_banked"].fillna(0) == 0].drop(columns="is_banked")
    pairs["gap_next"] = pairs["gap_next"].fillna(pairs["module_length"] - pairs["deadline"])
    pairs["deadline_frac"] = pairs["deadline"] / pairs["module_length"]
    pairs["season_j"] = pairs["code_presentation"].str.endswith("J").astype(np.int8)
    pairs = pairs.reset_index(drop=True)

    # table of all of a student's tasks in the same presentation (for history + workload)
    other = pairs[
        ["id_student", "code_presentation", "eid", "id_assessment", "deadline", "date_submitted", "score", "is_tma"]
    ].rename(
        columns={
            "eid": "eid_k",
            "id_assessment": "ida_k",
            "deadline": "d_k",
            "date_submitted": "sub_k",
            "score": "score_k",
            "is_tma": "tma_k",
        }
    )

    frames = []
    for H in horizons:
        s = pairs.copy()
        s["H"] = H
        s["t"] = s["deadline"] - H
        reg_ok = s["date_registration"].isna() | (s["date_registration"] <= s["t"])
        not_left = s["date_unregistration"].isna() | (s["date_unregistration"] > s["t"])
        not_done = s["date_submitted"].isna() | (s["date_submitted"] > s["t"])
        s = s[reg_ok & not_left & not_done].copy()
        s["y"] = (s["date_submitted"].notna() & (s["date_submitted"] <= s["deadline"])).astype(np.int8)
        s = s.reset_index(drop=True)
        s["sid"] = np.arange(len(s))

        # ---------- history & workload (other tasks of the same student, same presentation) ----------
        m = s[["sid", "id_student", "code_presentation", "eid", "id_assessment", "t", "is_tma"]].merge(
            other, on=["id_student", "code_presentation"], how="inner"
        )
        m = m[m["ida_k"] != m["id_assessment"]]
        same = m["eid_k"] == m["eid"]
        prior = same & (m["d_k"] <= m["t"])
        sub_known = m["sub_k"].notna() & (m["sub_k"] <= m["t"])
        on_time_k = m["sub_k"].notna() & (m["sub_k"] <= m["d_k"])
        same_type = prior & (m["tma_k"] == m["is_tma"])
        g = (
            pd.DataFrame(
                {
                    "sid": m["sid"],
                    "n_prior": prior,
                    "n_prior_same_type": same_type,
                    "n_prior_same_type_on_time": same_type & on_time_k,
                    "n_prior_on_time": prior & on_time_k,
                    "n_prior_late_known": prior & sub_known & ~on_time_k,
                    "n_prior_outstanding": prior & ~sub_known,
                    "n_future_done": same & (m["d_k"] > m["t"]) & sub_known,
                    "n_upcoming_14_all": (m["d_k"] > m["t"]) & (m["d_k"] <= m["t"] + 14) & ~sub_known,
                    "n_upcoming_7_all": (m["d_k"] > m["t"]) & (m["d_k"] <= m["t"] + 7) & ~sub_known,
                    "n_overdue_all": (m["d_k"] <= m["t"]) & ~sub_known,
                    "early_sum": np.where(prior & sub_known, m["d_k"] - m["sub_k"], 0.0),
                    "early_n": prior & sub_known,
                    "score_sum": np.where(
                        prior & sub_known & (m["d_k"] <= m["t"] - 14) & m["score_k"].notna(), m["score_k"], 0.0
                    ),
                    "score_n": prior & sub_known & (m["d_k"] <= m["t"] - 14) & m["score_k"].notna(),
                }
            )
            .groupby("sid")
            .sum()
        )
        last = m[prior].sort_values(["sid", "d_k"]).groupby("sid").tail(1)
        last_early = last[last["sub_k"] <= last["t"]]
        last_early = pd.Series((last_early["d_k"] - last_early["sub_k"]).to_numpy(), index=last_early["sid"].to_numpy())
        last = pd.Series((last["sub_k"] <= last["d_k"]).astype(float).to_numpy(), index=last["sid"].to_numpy())
        s = s.merge(g, left_on="sid", right_index=True, how="left")
        cnt_cols = list(g.columns)
        s[cnt_cols] = s[cnt_cols].fillna(0)
        s["prior_on_time_rate"] = np.where(s["n_prior"] > 0, s["n_prior_on_time"] / s["n_prior"].clip(lower=1), np.nan)
        s["prior_mean_days_early"] = np.where(s["early_n"] > 0, s["early_sum"] / s["early_n"].clip(lower=1), np.nan)
        s["prior_mean_score_14d"] = np.where(s["score_n"] > 0, s["score_sum"] / s["score_n"].clip(lower=1), np.nan)
        s["n_prior_scored_14d"] = s["score_n"]
        s["last_prior_on_time"] = s["sid"].map(last)
        s["last_prior_days_early"] = s["sid"].map(last_early)
        s["prior_same_type_on_time_rate"] = np.where(
            s["n_prior_same_type"] > 0, s["n_prior_same_type_on_time"] / s["n_prior_same_type"].clip(lower=1), np.nan
        )
        s = s.drop(columns=["early_sum", "early_n", "score_sum", "score_n", "n_prior_same_type_on_time"])

        # ---------- engagement from VLE clicks up to day t ----------
        e = s["eid"].to_numpy()
        ti = np.clip(s["t"].to_numpy().astype(int) - DMIN, -1, ND - 1)
        C, A = arr["tot"], arr["active"]
        big = np.full(len(s), ND, dtype=int)
        s["clicks_total"] = _wsum(C, e, ti, big)
        s["active_days_total"] = _wsum(A, e, ti, big)
        s["clicks_pre_start"] = np.where(ti >= 0, C[e, np.clip(min(-1 - DMIN, ND - 1), 0, None)], 0)
        s["clicks_pre_start"] = np.minimum(s["clicks_pre_start"], s["clicks_total"])
        for w in (3, 7, 14, 30):
            s[f"clicks_{w}"] = _wsum(C, e, ti, w)
        s["active_7"] = _wsum(A, e, ti, 7)
        s["active_14"] = _wsum(A, e, ti, 14)
        prev7 = _wsum(C, e, ti - 7, 7)
        s["trend_7"] = (s["clicks_7"] + 1) / (prev7 + 1)
        la = np.where(ti >= 0, arr["last_active"][e, np.clip(ti, 0, None)], -1)
        s["days_since_last"] = np.where(la >= 0, ti - la, np.nan)
        since = np.clip((s["t"] - (s["deadline"] - s["gap_prev"])).to_numpy().astype(int), 0, ND)
        s["clicks_since_prev_deadline"] = _wsum(C, e, ti, since)
        s["active_since_prev_deadline"] = _wsum(A, e, ti, since)
        for a_ in TOP_TYPES:
            s[f"{a_}_14"] = _wsum(arr[a_], e, ti, 14)
            s[f"{a_}_total"] = _wsum(arr[a_], e, ti, big)
        s["site_visits_7"] = _wsum(arr["sites"], e, ti, 7)
        s["site_visits_14"] = _wsum(arr["sites"], e, ti, 14)
        # recent effort vs the student's own weekly average so far (personal trend, not cohort)
        weeks_before = np.maximum(
            (s["t"].to_numpy() - np.minimum(s["date_registration"].fillna(0).to_numpy(), 0)) / 7 - 1, 1
        )
        s["self_rel_clicks_7"] = (s["clicks_7"] + 1) / ((s["clicks_total"] - s["clicks_7"]) / weeks_before + 1)
        s["days_since_registration"] = s["t"] - s["date_registration"]

        # cohort-relative engagement: compare with everyone registered in the same module-presentation at day t
        s["rel_clicks_7"] = np.nan
        s["pct_clicks_7"] = np.nan
        s["pct_clicks_14"] = np.nan
        s["pct_clicks_total"] = np.nan
        for (cm, cp, tt), idx in s.groupby(["code_module", "code_presentation", "t"]).groups.items():
            r = reg[(reg["code_module"] == cm) & (reg["code_presentation"] == cp)]
            r = r[
                (r["date_registration"].isna() | (r["date_registration"] <= tt))
                & (r["date_unregistration"].isna() | (r["date_unregistration"] > tt))
            ]
            ce = r["eid"].to_numpy()
            cti = np.full(len(ce), int(np.clip(tt - DMIN, -1, ND - 1)))
            for col, w in (("pct_clicks_7", 7), ("pct_clicks_14", 14), ("pct_clicks_total", ND)):
                coh = np.sort(_wsum(C, ce, cti, np.full(len(ce), w)))
                vals = s.loc[idx, col[len("pct_") :]].to_numpy()  # pct_clicks_7 -> clicks_7, etc.
                s.loc[idx, col] = np.searchsorted(coh, vals, side="right") / max(len(coh), 1)
                if w == 7:
                    s.loc[idx, "rel_clicks_7"] = vals / (np.median(coh) + 1)
        frames.append(s)
        if verbose:
            print(f"H={H:>2}: {len(s):,} samples, on-time rate {s['y'].mean():.3f}")

    df = pd.concat(frames, ignore_index=True)
    df = add_task_history(df)
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    return df


PRESENTATION_ORDER = ["2013B", "2013J", "2014B", "2014J"]


def add_task_history(df):
    """'How hard is this task?' — on-time rate of the same task slot (module, assessment order, horizon)
    in strictly EARLIER presentations only. App analogue: how often users finish this kind of task on time."""
    order = {p: i for i, p in enumerate(PRESENTATION_ORDER)}
    g = (
        df.groupby(["code_module", "a_order", "H", "code_presentation"], observed=True)["y"]
        .agg(["sum", "size"])
        .reset_index()
    )
    g["pi"] = g["code_presentation"].map(order)
    rows = []
    for _, r in g.iterrows():
        past = g[
            (g["code_module"] == r["code_module"])
            & (g["a_order"] == r["a_order"])
            & (g["H"] == r["H"])
            & (g["pi"] < r["pi"])
        ]
        n = past["size"].sum()
        rows.append(
            (r["code_module"], r["a_order"], r["H"], r["code_presentation"], past["sum"].sum() / n if n else np.nan, n)
        )
    h = pd.DataFrame(
        rows, columns=["code_module", "a_order", "H", "code_presentation", "task_hist_on_time", "task_hist_n"]
    )
    return df.merge(h, on=["code_module", "a_order", "H", "code_presentation"], how="left")
