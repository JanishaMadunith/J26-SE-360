import 'package:flutter/services.dart';

import '../data/models/sensing.dart';
import '../data/models/wire_enums.dart';

/// Dart side of the Kotlin MethodChannel `lk.sliit.j26se360/sensing` (see MainActivity.kt).
///
/// Only available in the UI isolate (the channel is registered by MainActivity). The background
/// worker reads the same data from the JSON-lines files the native side writes — see [FileBridge].
class NativeBridge {
  static const channel = MethodChannel('lk.sliit.j26se360/sensing');

  Future<bool> hasUsageAccess() async => await channel.invokeMethod<bool>('hasUsageAccess') ?? false;

  Future<void> openUsageAccessSettings() => channel.invokeMethod<void>('openUsageAccessSettings');

  Future<bool> isTypingServiceEnabled() async => await channel.invokeMethod<bool>('isTypingServiceEnabled') ?? false;

  Future<void> openAccessibilitySettings() => channel.invokeMethod<void>('openAccessibilitySettings');

  /// Foreground app sessions (category + salted package hash) in [from, to).
  Future<List<AppUsageSession>> getAppUsage(DateTime from, DateTime to) async {
    final raw = await channel.invokeListMethod<Map<Object?, Object?>>(
          'getAppUsage',
          {'fromMs': from.millisecondsSinceEpoch, 'toMs': to.millisecondsSinceEpoch},
        ) ??
        const [];
    return raw.map((m) => appUsageFromNative(_stringKeys(m))).toList();
  }

  /// Screen on / off / unlock events in [from, to).
  Future<List<ScreenEvent>> getScreenEvents(DateTime from, DateTime to) async {
    final raw = await channel.invokeListMethod<Map<Object?, Object?>>(
          'getScreenEvents',
          {'fromMs': from.millisecondsSinceEpoch, 'toMs': to.millisecondsSinceEpoch},
        ) ??
        const [];
    return raw.map((m) => screenEventFromNative(_stringKeys(m))).toList();
  }

  /// Returns AND removes the buffered keystroke timings (debug / inspection use).
  /// The uploader drains the same buffer through [FileBridge] instead.
  Future<List<KeystrokeEvent>> drainKeystrokes() async {
    final raw = await channel.invokeListMethod<Map<Object?, Object?>>('drainKeystrokes') ?? const [];
    return raw.map((m) => keystrokeFromNative(_stringKeys(m))).toList();
  }

  /// Asks the native side to write app-usage and screen events since its cursor into the JSONL files now
  /// (the native JobScheduler job also does this every 15 minutes).
  Future<void> snapshotNow() => channel.invokeMethod<void>('snapshotNow');

  /// Absolute path of the native sensing buffer directory (filesDir/sensing).
  Future<String?> getSensingDir() => channel.invokeMethod<String>('getSensingDir');

  /// IANA timezone id of the device, e.g. `Asia/Colombo`.
  Future<String?> getTimezone() async {
    try {
      return await channel.invokeMethod<String>('getTimezone');
    } on MissingPluginException {
      return null;
    }
  }

  static Map<String, dynamic> _stringKeys(Map<Object?, Object?> m) => m.map((k, v) => MapEntry('$k', v));
}

// ---- native JSON (epoch milliseconds) → contract models. Shared with FileBridge. ----

KeystrokeEvent keystrokeFromNative(Map<String, dynamic> m) => KeystrokeEvent(
      downAt: DateTime.fromMillisecondsSinceEpoch((m['down_at_ms'] as num).toInt()),
      holdMs: ((m['hold_ms'] as num?) ?? 0).toInt(),
      keyClass: KeyClass.fromWire(m['key_class']),
    );

AppUsageSession appUsageFromNative(Map<String, dynamic> m) => AppUsageSession(
      start: DateTime.fromMillisecondsSinceEpoch((m['start_ms'] as num).toInt()),
      end: DateTime.fromMillisecondsSinceEpoch((m['end_ms'] as num).toInt()),
      category: AppCategory.fromWire(m['category']),
      appHash: m['app_hash'] as String,
    );

ScreenEvent screenEventFromNative(Map<String, dynamic> m) => ScreenEvent(
      at: DateTime.fromMillisecondsSinceEpoch((m['at_ms'] as num).toInt()),
      type: ScreenEventType.fromWire(m['type']),
    );

ActivityTransition activityFromNative(Map<String, dynamic> m) => ActivityTransition(
      at: DateTime.fromMillisecondsSinceEpoch((m['at_ms'] as num).toInt()),
      activity: ActivityType.fromWire(m['activity']),
      entering: m['entering'] as bool,
    );
