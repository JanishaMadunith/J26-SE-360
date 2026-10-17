# Open team issues

Decisions the team still has to make or confirm. Discuss at the weekly meeting; when one is settled, record the outcome here (and as an ADR if it's an architecture decision).

| # | Issue | Status | Owner(s) |
|---|---|---|---|
| 1 | Number of affective states | **Open** | Jayatilake, all |
| 2 | Wellness forecast granularity (daily vs intraday) | **Resolved in contract — confirm** | Samarathunga, Kumarasinghe |
| 3 | M4 "Part B" task suggestions vs M3 scheduling | **Resolved in contract — confirm with supervisor** | Gunaweera, Kumarasinghe |
| 4 | Latency of chained M1 → M2 → M3 calls | **Resolved by design — measure** | Kumarasinghe |
| 5 | One shared pilot data-collection instrument | **Open** | Samarathunga, all |
| 6 | Ethics approval before the pilot | **Open — blocking the pilot** | All |
| 7 | Topic Assessment Form main-objective wording | **Open** | All |

## 1. Number of affective states

The Topic Assessment Form lists **five** states (stress, relaxed, happy, sad, anxiety). Some team documents use **three**. The contract (`AffectiveLabel` in `contracts/src/wellness_contracts/common.py`) currently uses the five from the form.

**Decide:** which set M1 will classify, given the data available for labels. If it changes, edit `AffectiveLabel` once and run `make contracts`. M2 and M4 consume the label and must handle the final set.

## 2. Wellness forecast granularity

Module 2's earlier spec forecast **daily** values. Module 3 needs **intraday** values to decide which slot suits which task.

**Resolution in the contract:** `WellnessForecast` returns a list of `WellnessSlot`s with `slot_minutes` granularity (default 60, allowed 15–240). Each slot has `mental_fatigue`, `focus`, `energy` and `burnout_risk` from 0 to 1. M2 can still model at daily level internally and expand to slots (e.g. with a circadian profile).

**Confirm:** Samarathunga agrees the slot output is feasible for M2's model and evaluation.

## 3. M4 "Part B" and the M3 boundary

A proposal slide showed M4 sending ranked tasks back to M3, which contradicts the one-way flow M1 → M2 → M3 → M4 and would create a loop.

**Resolution in the contract:** M4 returns `suggested_activities` (e.g. "15-minute walk"), which are **shown to the user only**. If the user accepts one, the app creates a normal task, and M3 schedules it on its next run. M4 decides *what* to suggest; M3 alone decides *when*. Data never flows from M4 straight into M3.

**Confirm:** supervisor sign-off on this boundary.

## 4. Latency

Sequential synchronous calls M1 → M2 → M3 could exceed each module's 2-second target.

**Resolution by design:**
- The pipeline runs in the **background** after uploads, not in the request path of the app.
- `POST /v1/schedule/regenerate` calls **only M3**, using the cached wellness forecast.
- Every call has a timeout, and failures fall back to the last stored result.

**Still to do:** measure per-module latency during the pilot and report it.

## 5. Shared pilot instrument

Running four separate data collections would exhaust participants. Samarathunga's pilot instrument could cover M1–M3, with the app's self-reports (`SelfReport`: mood, stress, energy, focus) providing labels for M1 and M2, and the task-event log providing M3's training data.

**Decide:** one combined protocol (duration, number of participants, check-in frequency) that serves all four modules.

## 6. Ethics approval

No real-participant data may be collected before SLIIT ethics approval. The application needs the data list in [data-and-privacy.md](data-and-privacy.md), the consent text, and the storage and retention plan.

**Decide:** who prepares the application and the target submission date. This blocks the pilot.

## 7. Topic Assessment Form wording

Section 8 of the form states the main objective as a food recommendation system only, rather than the full four-module framework. Check whether the wording should be corrected in later documents (proposal, progress reports, final thesis).
