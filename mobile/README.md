# Mobile app — `wellness_app` (Flutter + Kotlin, Android)

The phone app collects compact **raw signals**, uploads them to the core API, and shows the results of all
four modules: affective state (M1), energy/focus forecast (M2), adaptive schedule (M3) and meals (M4).
All analysis runs on the server (ADR-0004) — the app never runs a model.

> ⚠️ **Written without a Flutter SDK.** This skeleton was authored in an environment that could not reach
> pub.dev or the Flutter SDK, so it has **never been compiled or run**. Follow "First build" below and fix
> whatever the analyzer reports first — see [Unverified / check first](#unverified--check-first).

## Prerequisites

- Flutter **stable** (3.24 or newer, Dart ≥ 3.4) — `flutter doctor` must be green for Android
- Android Studio (Android SDK 34+, an emulator image or a USB cable)
- An **Android 8.0+ (API 26+) phone** — real sensors are needed for PPG, steps, light and typing rhythm.
  iOS is not supported (usage, typing and screen APIs are not available there).
- The backend running locally (`make up` from the repo root, core API on port 8000)

## First build

```bash
cd mobile/wellness_app

# 1. Generate the Gradle/platform files we do not commit. Existing files (lib/, AndroidManifest.xml,
#    Kotlin sources, res/xml, res/values/strings.xml, tests) are NOT overwritten.
flutter create --org lk.sliit.j26se360 --project-name wellness_app --platforms android .

# 2. Raise minSdk to 26: open android/app/build.gradle.kts (or build.gradle) and in defaultConfig set
#        minSdk = 26
#    (UsageStatsManager screen events and the native snapshot job assume API 26+.)

# 3. Dependencies, analysis, tests
flutter pub get          # if versions conflict: flutter pub upgrade --major-versions, then fix flagged APIs
flutter analyze
flutter test             # includes a contract test against ../../contracts/schemas

# 4. Run on a USB-connected phone (or emulator)
adb reverse tcp:8000 tcp:8000   # phone's 127.0.0.1:8000 → your computer's port 8000
flutter run
```

### Pointing the app at the backend

The server URL is editable in **Settings** (reachable from the login screen) and is used by the background
worker too.

| Setup | URL to use |
|---|---|
| USB phone or emulator with `adb reverse tcp:8000 tcp:8000` (recommended) | `http://127.0.0.1:8000` (default) |
| Emulator without adb reverse | `http://10.0.2.2:8000` |
| Phone on the same Wi-Fi as your computer | `http://<your computer's LAN IP>:8000` — start uvicorn with `--host 0.0.0.0` and allow it through the firewall |
| Deployed server | `https://…` (cleartext HTTP is only enabled for local development) |

**Settings → Save & test** calls `/health` and shows whether the server and modules answered.

## Granting the sensing permissions

Open **More → Settings & permissions**. Each row shows a ✓ or a *Grant* button.

1. **Usage access** (screen + app-category usage) — *Grant* opens *Settings → Usage access*; select
   **Wellness** and switch on *Permit usage access*.
2. **Typing rhythm service** — *Grant* opens *Settings → Accessibility*; open **Wellness typing rhythm**
   (sometimes under *Downloaded apps* / *Installed services*) and switch it on. On Android 13+ sideloaded
   apps may first need *App info → ⋮ → Allow restricted settings*.
3. **Physical activity** (steps + activity type) — standard runtime dialog.
4. **Notifications** — standard runtime dialog (Android 13+).
5. **Camera** — asked the first time you measure heart rate.

Returning to the app refreshes the statuses automatically.

## What is and is not collected (privacy)

| Signal | Sent to the server | Never collected |
|---|---|---|
| Heart rate (camera PPG) | mean R/G/B per frame (~30 numbers/s) during a 60 s measurement | photos, video |
| Typing rhythm | timestamp + key class (letter / digit / space / backspace / enter / punctuation / other) | the characters typed; anything from password fields; anything typed inside this app |
| App usage | start/end, Play-Store category, salted SHA-256 prefix of the package name (salt is per install and never uploaded) | app names, content, notifications |
| Screen | on / off / unlock times | — |
| Movement | activity transitions (still, walking, …) and step counts | location / GPS |
| Light | ambient lux readings | camera |
| Check-ins | mood, stress, energy, focus (1–5) and an optional note | — |

Rules enforced in code: the Pydantic contracts reject unknown fields (so a "typed_text" field could never be
accepted by the server), and the Kotlin service looks at an inserted character only to classify it.

## How data flows

```
Kotlin TypingTimingService ─┐                       ┌─ every 15 min: workmanager (Dart, background isolate)
Kotlin UsageSnapshotJob ────┼─► filesDir/sensing/*.jsonl ─► SensingRepository.collect() ─► SQLite upload queue
Dart ActivityCollector ─────┘   (rename-then-read hand-off)        + steps + light            │ (idempotent batch_id,
                                                                                               ▼  exponential back-off)
PPG screen / offline check-in ──────────────────────────────► queue ───────► POST /v1/sensing/batches (core API)
```

- The MethodChannel `lk.sliit.j26se360/sensing` only exists in the UI isolate (registered by `MainActivity`).
  The background worker therefore reads everything from the JSONL files, which the Kotlin side fills on its own
  (accessibility service + a framework `JobScheduler` job every ~15 minutes, persisted across reboots).
- Opening or resuming the app triggers a foreground sync (at most every 5 minutes); **Settings → Sync now** forces one.

## Code map and ownership

```
lib/
├── core/                 config, Dio client (+token refresh), secure token store, router, theme   — Dilsara (tech lead)
├── data/models/          one Dart class per contract (snake_case JSON = contracts/schemas)        — all (change with contracts)
├── data/api/apis.dart    one class per core-API area                                               — Dilsara
├── sensing/              collectors, file bridge, batch builder, upload queue, background sync     — Jayatilake (M1) + Dilsara
├── features/
│   ├── auth/, onboarding_consent/, shell/, settings/                                               — Dilsara
│   ├── ppg_measure/, check_in/                                                                     — Jayatilake (M1)
│   ├── home_dashboard/   (affective card + energy/focus forecast)                                  — Jayatilake (M1) / Samarathunga (M2)
│   ├── tasks/, schedule/                                                                           — Kumarasinghe (M3)
│   └── meals/                                                                                      — Gunaweera (M4)
android/app/src/main/kotlin/lk/sliit/j26se360/wellness_app/
├── MainActivity.kt        MethodChannel handler                                                    — Dilsara
├── UsageCollector.kt      UsageStatsManager → sessions + screen events, category + salted hash     — Jayatilake (M1)
├── TypingTimingService.kt accessibility service: key timing + class only                           — Jayatilake (M1)
├── UsageSnapshotJob.kt    15-min JobScheduler snapshot into the JSONL buffers                      — Jayatilake (M1)
└── SensingFiles.kt        JSONL buffer helper (rename-then-read)                                   — Jayatilake (M1)
test/                      contract test vs ../../contracts/schemas, model round-trips, PPG maths, batch builder, upload queue
```

State management: **Riverpod** (`FutureProvider` per read model; screens call `refreshAll(ref)` after changes).
Navigation: **go_router** with an auth/consent redirect. HTTP: **Dio** with a queued interceptor that refreshes
the access token once on 401.

When a contract changes: edit `contracts/`, run `make contracts`, update the Dart model, run `flutter test`
(the contract test fails until both sides agree).

## Unverified / check first

Everything below was written from memory of the package APIs, without compiling:

1. **Package versions** in `pubspec.yaml` (riverpod 2.6, go_router 14, workmanager 0.5, camera 0.11,
   light 4, flutter_activity_recognition 4, permission_handler 11, …). If `pub get` resolves newer majors,
   expect small API renames.
2. **workmanager** — `Workmanager().initialize(callbackDispatcher)` and `registerPeriodicTask(name, name,
   frequency:)`. Newer majors renamed some enums/parameters (`existingWorkPolicy`, `isInDebugMode`).
3. **flutter_activity_recognition** — `FlutterActivityRecognition.instance`, `checkPermission()`,
   `requestPermission()`, `ActivityPermission.GRANTED`, `activityStream`, `Activity.type`. The enum constant
   casing may differ by version (mapping uses `.name`, so only the permission constant is at risk).
4. **light** — `Light().lightSensorStream` (`Stream<int>`).
5. **pedometer** — `Pedometer.stepCountStream` / `StepCount.steps`.
6. **camera** — YUV420 image stream: plane order Y/U/V, `bytesPerRow`, `bytesPerPixel`; torch via
   `setFlashMode(FlashMode.torch)` while streaming works on most but not all devices.
7. **Riverpod 2.x APIs** used: `AsyncNotifier`, `NotifierProvider`, `AsyncValue.valueOrNull`, `hasValue`.
   Riverpod 3 renamed `valueOrNull` → `value`.
8. **Kotlin**: `AccessibilityService` declaration (`exported="true"` + `BIND_ACCESSIBILITY_SERVICE`),
   `UsageEvents` numeric event types (1, 2, 15, 16, 18), `JobScheduler` periodic job, `QUERY_ALL_PACKAGES`
   (fine for a research APK, not allowed on the Play Store).
9. **Path hand-off**: Dart reads `filesDir/sensing` using the path cached from `getSensingDir`; the fallback
   assumes `getApplicationSupportDirectory()` == `filesDir` on Android.
10. **Gradle**: only `minSdk = 26` should need changing after `flutter create`. If the camera or permission
    plugins ask for a higher `compileSdk`, set it in `android/app/build.gradle(.kts)`.
11. `flutter analyze` treats lint *infos* as failures in CI — run `dart fix --apply` and `dart format .` once
    after the first successful build.
