"""Synthetic training data for M2 (wellness forecast).

One 'request' = one user-day at a random time of day.
Each request has 24 hourly slots with ground-truth scores in 0-1.
All formulas are ILLUSTRATIVE assumptions, so document them as such.
"""
import os
import pathlib

import numpy as np
import pandas as pd

from .features import slot_features


def sig(z):
    return 1 / (1 + np.exp(-z))

MISSING_COLS = ["sleep_h", "wake_hour", "stress_last", "stress_mean7",
                "stress_trend", "aff_conf", "steps_24h"]


def simulate(n_users=150, days=40, seed=0, missing_rate=0.12):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        base_sleep = rng.normal(7, 0.8)      # personal sleep habit
        sens = rng.uniform(0.7, 1.3)         # sensitivity to sleep debt
        chrono = rng.normal(0, 1)            # early bird vs night owl
        stress_hist = list(rng.uniform(0.2, 0.5, 7))
        overdue = 0

        for d in range(days):
            sleep = np.clip(rng.normal(base_sleep, 1.0), 3.5, 10)
            wake = np.clip(7 + chrono + 0.3 * (sleep - 7) + rng.normal(0, 0.5), 4.5, 11)
            tasks = rng.poisson(4)
            mins = max(0.0, tasks * rng.normal(50, 15))
            overdue = max(0, overdue + rng.poisson(0.8) - rng.poisson(1.0))
            debt = max(0.0, 7.5 - sleep)

            stress = np.clip(0.25 + 0.03 * tasks + 0.04 * overdue + 0.04 * debt
                             + rng.normal(0, 0.08), 0, 1)
            stress_hist = [*stress_hist, stress][-7:]
            s7 = float(np.mean(stress_hist))
            steps = np.clip(rng.normal(6500, 3000), 500, 20000)
            asof = rng.uniform(6, 22)        # time of day the request is made

            load = min(1.0, mins / 480 + 0.08 * overdue)
            burnout = sig(-3 + 3.0 * s7 + 0.25 * debt + 1.0 * load + rng.normal(0, 0.15))

            req = {
                "sleep_h": sleep, "wake_hour": wake, "stress_last": stress, "stress_mean7": s7,
                "stress_trend": stress - s7, "aff_conf": rng.uniform(0.6, 0.95),
                "steps_24h": steps, "tasks_due_24h": tasks, "minutes_due_24h": mins,
                "tasks_overdue": overdue, "tasks_pending": tasks + overdue + rng.poisson(2),
                "asof_hour": asof,
            }

            for k in range(24):
                sf = slot_features(asof, wake, k)
                h = (asof + k) % 24
                # daily rhythm: late-morning peak, post-lunch dip
                alert = 0.5 + 0.25 * np.cos(2 * np.pi * (h - 11) / 24) \
                        - 0.12 * np.exp(-((h - 14.5) / 1.3) ** 2)
                fat = sig(-2 + 0.11 * sf["hours_awake"] + 0.3 * sens * debt + 1.1 * load
                          + 1.5 * stress - 2 * (alert - 0.5) + rng.normal(0, 0.15))
                focus = np.clip(0.35 + 0.9 * alert - 0.5 * fat - 0.25 * stress
                                - 0.1 * load + rng.normal(0, 0.05), 0, 1)
                energy = np.clip(0.2 + 0.9 * alert - 0.45 * fat + 1e-5 * steps
                                 + rng.normal(0, 0.05), 0, 1)
                rows.append(dict(
                    user_id=u, day=d, **req, **sf,
                    mental_fatigue=fat, focus=focus, energy=energy,
                    burnout_risk=np.clip(burnout + rng.normal(0, 0.02), 0, 1),
                ))

    df = pd.DataFrame(rows)

    # Blank inputs per REQUEST (all 24 slots of a user-day together),
    # so the model learns to cope with missing data like the real API will.
    req_idx = df.groupby(["user_id", "day"]).ngroup().to_numpy()
    n_req = req_idx.max() + 1
    for c in MISSING_COLS:
        df.loc[(rng.random(n_req) < missing_rate)[req_idx], c] = np.nan
    df.loc[df["wake_hour"].isna(), "hours_awake"] = np.nan  # derived from wake_hour
    return df


if __name__ == "__main__":
    out = pathlib.Path(os.getenv("DATA_DIR", "../../../07-datasets")) / "m2_synthetic.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = simulate()
    df.to_csv(out, index=False)
    print("shape:", df.shape, "->", out.resolve())
    print(df[["mental_fatigue", "focus", "energy", "burnout_risk"]].describe().T[["mean", "min", "max"]])
    print("missing %:\n", (df.isna().mean() * 100).round(1)[lambda s: s > 0])
