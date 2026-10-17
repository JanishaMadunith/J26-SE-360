import 'package:pedometer/pedometer.dart' as pedo;
import 'package:shared_preferences/shared_preferences.dart';

import '../../core/config/app_config.dart';
import '../../data/models/sensing.dart';
import '../batch_builder.dart';

/// Turns the device's cumulative step counter into a [StepCount] for the period since the last reading.
/// Needs the ACTIVITY_RECOGNITION runtime permission on Android 10+.
class StepsCollector {
  Future<StepCount?> readSince(SharedPreferences prefs, DateTime now) async {
    int current;
    try {
      final event = await pedo.Pedometer.stepCountStream.first.timeout(const Duration(seconds: 4));
      current = event.steps;
    } catch (_) {
      return null; // no sensor, no permission, or no reading in time
    }
    final previous = prefs.getInt(AppConfig.kLastSteps);
    final previousAtMs = prefs.getInt(AppConfig.kLastStepsAt);
    await prefs.setInt(AppConfig.kLastSteps, current);
    await prefs.setInt(AppConfig.kLastStepsAt, now.millisecondsSinceEpoch);
    if (previous == null || previousAtMs == null) return null; // first reading = baseline
    final start = DateTime.fromMillisecondsSinceEpoch(previousAtMs);
    if (!now.isAfter(start)) return null;
    return StepCount(start: start, end: now, steps: BatchBuilder.stepDelta(previous: previous, current: current));
  }
}
