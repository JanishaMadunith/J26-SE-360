# ADR-0001: Flutter app, Android-first, with Kotlin for native sensing

**Status:** Accepted — 17 Oct 2026

## Context

The app must collect smartphone signals without wearables: camera-based PPG, typing rhythm, screen events, app usage, physical activity, ambient light and self-reports. It also needs ordinary screens: login, tasks, schedule, insights, meals.

On iPhone, the key signals are not available to ordinary apps: there is no general app-usage history, no system-wide typing timings, and background collection is tightly restricted. Android exposes `UsageStatsManager`, accessibility events, activity recognition and sensors. The tech lead already knows Flutter.

## Decision

- Build the app in **Flutter** (Dart 3), targeting **Android** (minSdk 26). iOS is out of scope.
- Use **Kotlin** platform code for signals that Flutter plugins don't cover well: app-usage sessions (with category and hashed name), screen events, and the typing-timing accessibility service. They're exposed to Dart through a method channel.
- Use maintained Flutter plugins where they fit: `camera` (PPG), `pedometer`, `light`, `flutter_activity_recognition`, `permission_handler`, `workmanager` (15-minute background upload), `sqflite` (offline queue).
- App architecture: Riverpod for state, go_router for navigation, dio for HTTP, flutter_secure_storage for tokens. Feature-first folders under `lib/features/`, each mapped to a module owner.

## Alternatives considered

| Option | Why not |
|---|---|
| React Native (Expo dev build) + Kotlin | Viable, but sensing still needs native Kotlin modules, so no real saving. The team has more Flutter experience. |
| Native Android only (Kotlin + Compose) | Best sensor access, but more UI code to write and maintain. Most of the team would need to learn Compose. |
| Cross-platform with iOS | iOS cannot provide app-usage or typing signals, so M1 would not work there. Supporting it doubles testing effort for no research benefit. |

## Consequences

- One UI codebase; native complexity is limited to a few Kotlin classes.
- Results apply to Android users only — state this as a limitation in the thesis.
- Physical Android phones are needed for development and the pilot. Emulators can't provide camera PPG or real usage data.
- Accessibility-service-based typing collection is acceptable for a sideloaded research APK, but would need review before any Play Store release.
