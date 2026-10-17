import '../data/models/sensing.dart';

/// Pure function that assembles a [SensorBatch] (unit-tested in test/batch_builder_test.dart).
///
/// The collection window is widened to include every event's timestamp, so `collected_from <=
/// collected_to` always holds and the server can trust the window when slicing data for Module 1.
class BatchBuilder {
  static SensorBatch build({
    required String batchId,
    required String deviceId,
    required String appVersion,
    required String timezone,
    required DateTime from,
    required DateTime to,
    List<PPGRecording> ppg = const [],
    List<KeystrokeEvent> keystrokes = const [],
    List<ScreenEvent> screenEvents = const [],
    List<AppUsageSession> appUsage = const [],
    List<ActivityTransition> activity = const [],
    List<StepCount> steps = const [],
    List<LightReading> light = const [],
    List<SelfReport> selfReports = const [],
  }) {
    final times = <DateTime>[
      from,
      to,
      for (final r in ppg) r.startedAt,
      for (final k in keystrokes) k.downAt,
      for (final s in screenEvents) s.at,
      for (final a in appUsage) ...[a.start, a.end],
      for (final a in activity) a.at,
      for (final s in steps) ...[s.start, s.end],
      for (final l in light) l.at,
      for (final s in selfReports) s.at,
    ];
    final earliest = times.reduce((a, b) => a.isBefore(b) ? a : b);
    final latest = times.reduce((a, b) => a.isAfter(b) ? a : b);

    int byTime<T>(DateTime Function(T) at, T a, T b) => at(a).compareTo(at(b));

    return SensorBatch(
      batchId: batchId,
      deviceId: deviceId,
      appVersion: appVersion,
      timezone: timezone,
      collectedFrom: earliest,
      collectedTo: latest,
      ppg: ppg,
      keystrokes: [...keystrokes]..sort((a, b) => byTime<KeystrokeEvent>((e) => e.downAt, a, b)),
      screenEvents: [...screenEvents]..sort((a, b) => byTime<ScreenEvent>((e) => e.at, a, b)),
      appUsage: [...appUsage]
        ..removeWhere((s) => !s.end.isAfter(s.start))
        ..sort((a, b) => byTime<AppUsageSession>((e) => e.start, a, b)),
      activity: [...activity]..sort((a, b) => byTime<ActivityTransition>((e) => e.at, a, b)),
      steps: steps.where((s) => s.steps >= 0 && !s.end.isBefore(s.start)).toList(),
      light: light.where((l) => l.lux >= 0).toList(),
      selfReports: selfReports,
    );
  }

  /// Steps taken since the previous reading of the device's cumulative step counter.
  /// The counter resets to 0 on reboot, so a smaller value means "steps since reboot".
  static int stepDelta({required int previous, required int current}) =>
      current >= previous ? current - previous : current;
}
