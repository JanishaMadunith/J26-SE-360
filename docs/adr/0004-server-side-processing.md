# ADR-0004: All feature extraction and models run on the server

**Status:** Accepted — 17 Oct 2026

## Context

Signals start on the phone, but turning them into features (heart rate and HRV from PPG, typing speed and error rate, sleep estimates) and running models could happen on the phone, on the server, or split between them.

On-phone processing saves bandwidth and keeps raw data local. However, it means implementing and testing feature code twice (Dart/Kotlin for the app, Python for training) and converting models for mobile. XGBoost and OR-Tools don't run on phones easily.

## Decision

**The phone only collects and uploads. All feature extraction and every model run on the server**, in the module services, in Python — the same code used for training.

To keep this practical, the phone sends **compact raw signals, never raw media or content**:

| Signal | Sent as | Not sent |
|---|---|---|
| Camera PPG | Per-frame mean R/G/B values (~30 numbers per second) | Video frames |
| Typing | Key-down time, hold time, key class | Characters |
| App usage | Sessions with category + salted hash | App names, content |
| Screen, activity, steps, light | Events and readings | — |

Full details: [data-and-privacy.md](../data-and-privacy.md).

## Alternatives considered

| Option | Why not |
|---|---|
| Phone extracts features, server predicts | Lowest bandwidth and most private, but feature code would exist twice (training in Python, inference in Dart/Kotlin) and could drift — a common source of silent accuracy loss. |
| Models on the phone (TFLite) | Works offline, but model conversion is extra work for every member, and XGBoost and OR-Tools are hard to run on-device. |
| Upload raw media (video, full text) | Simplest, but unacceptable for privacy and ethics, and far too heavy on data and battery. |

## Consequences

**Benefits**

- One implementation of every feature, shared by training and serving — no train/serve skew.
- Members change their feature extraction or model without releasing a new app.
- Raw signals are stored, so features can be recomputed later for new experiments.

**Costs, stated honestly**

- **Bandwidth and battery.** Raw signals are bigger than features. A 60-second PPG measurement is ~1,800 frames × 4 numbers (about 50–80 KB of JSON). A day of keystroke timings for a heavy typist can be a few MB. Mitigations: batch uploads every ~15 minutes, gzip compression, upload on Wi-Fi or while charging when the queue is large, and an offline queue so nothing is lost.
- **Privacy.** The server holds more detailed data than with on-phone features. Mitigations: no content is ever captured (enforced by the contracts — keystroke events have no field for characters, unknown fields are rejected), app names are hashed on the phone, everything is behind authentication on our own database, and research exports are pseudonymised.
- **No offline insights.** Without a connection, the app can show the last downloaded schedule and meals but can't compute new ones.
- **Server load.** All computation is central. Fine at pilot scale (tens of users). Revisit if scaling up.

If bandwidth or ethics review becomes a problem, the fallback is to move feature extraction for one signal (most likely PPG) onto the phone, keeping the server-side code as the reference implementation. That would be recorded in a new ADR.
