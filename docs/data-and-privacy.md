# Data and privacy

What the app collects, what it never collects, and how the data is protected. Keep this page accurate — it is the basis for the ethics application and for the consent screen in the app. **Any new signal must be added here in the same PR that starts collecting it.**

## Principles

1. **Collect raw signals, not content.** The phone records timings, counts and numeric readings — never what the user types, sees or says.
2. **Process on the server.** All feature extraction and all models run server-side ([ADR-0004](adr/0004-server-side-processing.md)), so the phone sends compact raw signals rather than media.
3. **Consent first.** Nothing is collected until the user accepts the current consent version in the app.
4. **Pseudonymous research data.** Analysis exports use participant codes, never emails or names.
5. **Data never enters git.** Not in the repo, issues, PRs or chat.

## What is collected

| Signal | Exactly what the phone sends | Never sent | Used by |
|---|---|---|---|
| Camera heart rate (PPG) | During a measurement the user starts (finger over camera, flash on): the **mean red, green and blue value of each frame** with its timestamp, ~30 values per second | Camera images or video | M1 (heart rate, HRV) |
| Typing rhythm | For each key press: time pressed, how long it was held, and its **class** (letter, digit, space, backspace, enter, punctuation, other) | **The characters typed**, the app typed in, or anything from password fields | M1 |
| Screen use | Screen on, screen off and unlock events with timestamps | Screen contents | M1, M2 (sleep estimate) |
| App use | Foreground sessions: start, end, app **category** (social, game, productivity, …) and a **salted hash** of the app's package name | Plain app names, anything inside apps, notifications | M1 |
| Physical activity | Activity transitions (still, walking, running, cycling, vehicle) and step counts per interval | GPS or any location | M1, M2 |
| Ambient light | Light level (lux) readings at intervals | Camera | M2 (sleep estimate) |
| Self-reports | Optional check-ins: mood, stress, energy, focus (1–5) and an optional short note (≤ 280 characters) | — | Ground-truth labels for M1 and M2 |
| Tasks | Titles, categories, estimates, deadlines, priorities the user enters, and every status change | — | M3 |
| Diet profile | Dietary preferences, allergies, dislikes, health conditions the user chooses to enter | — | M4 |
| Account | Email, display name, timezone, hashed password, consent version and time | Plain-text password | Core API |

The contract enforces this. `SensorBatch` rejects unknown fields, so a field like `text_typed` cannot be uploaded by accident. Keystroke events have no field that could hold a character.

### How the typing signal stays content-free

On Android, typing timings come from an accessibility service. The service checks the class of each change (letter, digit, …) on the phone, keeps only that class and the timing, and immediately discards the text. Password fields are ignored entirely. The service is off until the user enables it in system settings, and the app explains why before sending them there.

### App hashing

The package name is hashed on the phone with SHA-256 and a random salt created at install time, and only a prefix is sent. The server can tell "the same app as before" for one user, but cannot recover which app it was or link apps across users.

## Storage and security

- All data is stored in the project's own PostgreSQL database ([ADR-0003](adr/0003-postgresql-and-own-auth.md)), never on a third-party analytics platform.
- Passwords are hashed with bcrypt. Access uses short-lived JWT access tokens (60 min) and refresh tokens (30 days).
- In any deployment outside a laptop: HTTPS only, a strong `JWT_SECRET`, a database password that is not the development default, and database access limited to the core API.
- Secrets live in `.env`, which is gitignored. Never commit `.env`, keystores or tokens.
- Module services never touch the database. They only receive the minimum data needed for one prediction.

## Consent flow (in the app)

1. On first launch, the app shows the information sheet and consent screen (version-numbered, e.g. `1.0`).
2. Collection starts only after the user accepts. The app calls `POST /v1/me/consent {version}`.
3. Each permission (camera, usage access, accessibility, activity recognition) is requested separately, with a one-line explanation of why.
4. If the consent text changes, the version increases and the app asks again before collecting.
5. Users can pause collection at any time in Settings. On withdrawal, collection stops and their data is deleted on request.

## Retention

- Raw signals are kept only for the study period plus analysis, then deleted. Derived features and module outputs may be kept in pseudonymised form for the thesis.
- Development databases on laptops contain **only simulated data** (`make simulate`), never participant data.

## Research export

For analysis, data is exported from the database to `../07-datasets/pilot/` — **outside the repo** — with:

- user IDs replaced by participant codes (P01, P02, …);
- email, display name and free-text notes removed;
- timestamps kept (needed for time-series models) but no location data exists to remove.

The export script will live in `scripts/` (planned). Until then, ask the tech lead before running any manual export.

## Ethics

- Data collection from participants (the pilot study) starts **only after** SLIIT ethics approval. See [open-issues.md](open-issues.md).
- Use one shared pilot data-collection instrument for the whole team where possible, so participants complete one study, not four.
- Participants must be adults who gave informed consent. They can withdraw without giving a reason.
