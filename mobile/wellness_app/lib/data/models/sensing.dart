/// Raw-signal contracts sent from the phone (`contracts/src/wellness_contracts/sensing.py`).
///
/// Privacy rules enforced by these types: no video frames (PPG is per-frame colour means), no typed
/// characters (keystrokes carry timing + key class only), no package names (salted hash + category).
library;

import '../../core/utils/json.dart';
import 'wire_enums.dart';

/// One camera heart-rate measurement. Arrays are column-wise and equal length.
class PPGRecording {
  PPGRecording({
    required this.startedAt,
    required this.fps,
    required this.tMs,
    required this.red,
    required this.green,
    required this.blue,
    this.torchOn = true,
  }) : assert(
          tMs.length == red.length && red.length == green.length && green.length == blue.length,
          't_ms, red, green and blue must have the same length',
        );

  factory PPGRecording.fromJson(JsonMap json) => PPGRecording(
        startedAt: parseDate(json['started_at']),
        fps: toDouble(json['fps']),
        tMs: parseInts(json['t_ms']),
        red: parseDoubles(json['red']),
        green: parseDoubles(json['green']),
        blue: parseDoubles(json['blue']),
        torchOn: (json['torch_on'] as bool?) ?? true,
      );

  /// Contract limit: at most 7,200 frames per recording.
  static const maxFrames = 7200;

  final DateTime startedAt;
  final double fps;
  final List<int> tMs;
  final List<double> red;
  final List<double> green;
  final List<double> blue;
  final bool torchOn;

  JsonMap toJson() => {
        'started_at': dateToJson(startedAt),
        'fps': fps,
        't_ms': tMs,
        'red': red,
        'green': green,
        'blue': blue,
        'torch_on': torchOn,
      };
}

/// Timing of one key press — never the character.
class KeystrokeEvent {
  const KeystrokeEvent({required this.downAt, required this.holdMs, required this.keyClass});

  factory KeystrokeEvent.fromJson(JsonMap json) => KeystrokeEvent(
        downAt: parseDate(json['down_at']),
        holdMs: toInt(json['hold_ms']),
        keyClass: KeyClass.fromWire(json['key_class']),
      );

  final DateTime downAt;

  /// 0 when unknown (Android accessibility events have no key-up).
  final int holdMs;
  final KeyClass keyClass;

  JsonMap toJson() => {'down_at': dateToJson(downAt), 'hold_ms': holdMs, 'key_class': keyClass.wire};
}

class ScreenEvent {
  const ScreenEvent({required this.at, required this.type});

  factory ScreenEvent.fromJson(JsonMap json) =>
      ScreenEvent(at: parseDate(json['at']), type: ScreenEventType.fromWire(json['type']));

  final DateTime at;
  final ScreenEventType type;

  JsonMap toJson() => {'at': dateToJson(at), 'type': type.wire};
}

class AppUsageSession {
  const AppUsageSession({
    required this.start,
    required this.end,
    required this.category,
    required this.appHash,
  });

  factory AppUsageSession.fromJson(JsonMap json) => AppUsageSession(
        start: parseDate(json['start']),
        end: parseDate(json['end']),
        category: AppCategory.fromWire(json['category']),
        appHash: json['app_hash'] as String,
      );

  final DateTime start;
  final DateTime end;
  final AppCategory category;

  /// Salted SHA-256 of the package name (first 16 hex chars) — computed natively.
  final String appHash;

  JsonMap toJson() => {
        'start': dateToJson(start),
        'end': dateToJson(end),
        'category': category.wire,
        'app_hash': appHash,
      };
}

class ActivityTransition {
  const ActivityTransition({required this.at, required this.activity, required this.entering});

  factory ActivityTransition.fromJson(JsonMap json) => ActivityTransition(
        at: parseDate(json['at']),
        activity: ActivityType.fromWire(json['activity']),
        entering: json['entering'] as bool,
      );

  final DateTime at;
  final ActivityType activity;
  final bool entering;

  JsonMap toJson() => {'at': dateToJson(at), 'activity': activity.wire, 'entering': entering};
}

class StepCount {
  const StepCount({required this.start, required this.end, required this.steps});

  factory StepCount.fromJson(JsonMap json) =>
      StepCount(start: parseDate(json['start']), end: parseDate(json['end']), steps: toInt(json['steps']));

  final DateTime start;
  final DateTime end;
  final int steps;

  JsonMap toJson() => {'start': dateToJson(start), 'end': dateToJson(end), 'steps': steps};
}

class LightReading {
  const LightReading({required this.at, required this.lux});

  factory LightReading.fromJson(JsonMap json) => LightReading(at: parseDate(json['at']), lux: toDouble(json['lux']));

  final DateTime at;
  final double lux;

  JsonMap toJson() => {'at': dateToJson(at), 'lux': lux};
}

/// In-app check-in (1–5 scales). Ground-truth labels for Modules 1 and 2.
class SelfReport {
  const SelfReport({required this.at, this.mood, this.stress, this.energy, this.focus, this.note});

  factory SelfReport.fromJson(JsonMap json) => SelfReport(
        at: parseDate(json['at']),
        mood: toIntOrNull(json['mood']),
        stress: toIntOrNull(json['stress']),
        energy: toIntOrNull(json['energy']),
        focus: toIntOrNull(json['focus']),
        note: json['note'] as String?,
      );

  final DateTime at;
  final int? mood;
  final int? stress;
  final int? energy;
  final int? focus;
  final String? note;

  JsonMap toJson() => {
        'at': dateToJson(at),
        'mood': mood,
        'stress': stress,
        'energy': energy,
        'focus': focus,
        'note': note,
      };
}

/// One upload. `batchId` (UUID v4) makes re-sending the same batch safe.
class SensorBatch {
  const SensorBatch({
    required this.batchId,
    required this.deviceId,
    required this.appVersion,
    required this.timezone,
    required this.collectedFrom,
    required this.collectedTo,
    this.ppg = const [],
    this.keystrokes = const [],
    this.screenEvents = const [],
    this.appUsage = const [],
    this.activity = const [],
    this.steps = const [],
    this.light = const [],
    this.selfReports = const [],
  });

  factory SensorBatch.fromJson(JsonMap json) => SensorBatch(
        batchId: json['batch_id'] as String,
        deviceId: json['device_id'] as String,
        appVersion: json['app_version'] as String,
        timezone: json['timezone'] as String,
        collectedFrom: parseDate(json['collected_from']),
        collectedTo: parseDate(json['collected_to']),
        ppg: parseList(json['ppg'], PPGRecording.fromJson),
        keystrokes: parseList(json['keystrokes'], KeystrokeEvent.fromJson),
        screenEvents: parseList(json['screen_events'], ScreenEvent.fromJson),
        appUsage: parseList(json['app_usage'], AppUsageSession.fromJson),
        activity: parseList(json['activity'], ActivityTransition.fromJson),
        steps: parseList(json['steps'], StepCount.fromJson),
        light: parseList(json['light'], LightReading.fromJson),
        selfReports: parseList(json['self_reports'], SelfReport.fromJson),
      );

  final String batchId;
  final String deviceId;
  final String appVersion;
  final String timezone;
  final DateTime collectedFrom;
  final DateTime collectedTo;
  final List<PPGRecording> ppg;
  final List<KeystrokeEvent> keystrokes;
  final List<ScreenEvent> screenEvents;
  final List<AppUsageSession> appUsage;
  final List<ActivityTransition> activity;
  final List<StepCount> steps;
  final List<LightReading> light;
  final List<SelfReport> selfReports;

  bool get isEmpty =>
      ppg.isEmpty &&
      keystrokes.isEmpty &&
      screenEvents.isEmpty &&
      appUsage.isEmpty &&
      activity.isEmpty &&
      steps.isEmpty &&
      light.isEmpty &&
      selfReports.isEmpty;

  Map<String, int> get counts => {
        'ppg': ppg.length,
        'keystrokes': keystrokes.length,
        'screen_events': screenEvents.length,
        'app_usage': appUsage.length,
        'activity': activity.length,
        'steps': steps.length,
        'light': light.length,
        'self_reports': selfReports.length,
      };

  JsonMap toJson() => {
        'batch_id': batchId,
        'device_id': deviceId,
        'app_version': appVersion,
        'timezone': timezone,
        'collected_from': dateToJson(collectedFrom),
        'collected_to': dateToJson(collectedTo),
        'ppg': ppg.map((e) => e.toJson()).toList(),
        'keystrokes': keystrokes.map((e) => e.toJson()).toList(),
        'screen_events': screenEvents.map((e) => e.toJson()).toList(),
        'app_usage': appUsage.map((e) => e.toJson()).toList(),
        'activity': activity.map((e) => e.toJson()).toList(),
        'steps': steps.map((e) => e.toJson()).toList(),
        'light': light.map((e) => e.toJson()).toList(),
        'self_reports': selfReports.map((e) => e.toJson()).toList(),
      };
}

/// `SensorBatchAck` (202 response of `POST /v1/sensing/batches`).
class SensorBatchAck {
  const SensorBatchAck({required this.batchId, required this.accepted, this.duplicate = false, this.counts = const {}});

  factory SensorBatchAck.fromJson(JsonMap json) => SensorBatchAck(
        batchId: json['batch_id'] as String,
        accepted: json['accepted'] as bool,
        duplicate: (json['duplicate'] as bool?) ?? false,
        counts: ((json['counts'] as JsonMap?) ?? const <String, dynamic>{}).map((k, v) => MapEntry(k, toInt(v))),
      );

  final String batchId;
  final bool accepted;
  final bool duplicate;
  final Map<String, int> counts;

  JsonMap toJson() => {'batch_id': batchId, 'accepted': accepted, 'duplicate': duplicate, 'counts': counts};
}
