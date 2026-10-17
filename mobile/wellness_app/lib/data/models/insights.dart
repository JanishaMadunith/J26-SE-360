import '../../core/utils/json.dart';
import 'model_info.dart';
import 'wire_enums.dart';

/// `AffectiveStateResult` (Module 1 output).
class AffectiveStateResult {
  const AffectiveStateResult({
    required this.userId,
    required this.windowStart,
    required this.windowEnd,
    required this.label,
    required this.probabilities,
    required this.confidence,
    required this.model,
    this.features = const {},
  });

  factory AffectiveStateResult.fromJson(JsonMap json) => AffectiveStateResult(
        userId: json['user_id'] as String,
        windowStart: parseDate(json['window_start']),
        windowEnd: parseDate(json['window_end']),
        label: AffectiveLabel.fromWire(json['label']),
        probabilities: ((json['probabilities'] as JsonMap?) ?? const <String, dynamic>{})
            .map((k, v) => MapEntry(k, toDouble(v))),
        confidence: toDouble(json['confidence']),
        features: ((json['features'] as JsonMap?) ?? const <String, dynamic>{})
            .map((k, v) => MapEntry(k, toDoubleOrNull(v))),
        model: ModelInfo.fromJson(json['model'] as JsonMap),
      );

  final String userId;
  final DateTime windowStart;
  final DateTime windowEnd;
  final AffectiveLabel label;

  /// Keyed by the label's wire value (e.g. `"stress"`).
  final Map<String, double> probabilities;
  final double confidence;

  /// Features Module 1 extracted on the server, e.g. `hr_bpm`, `hrv_rmssd_ms`.
  final Map<String, double?> features;
  final ModelInfo model;

  JsonMap toJson() => {
        'user_id': userId,
        'window_start': dateToJson(windowStart),
        'window_end': dateToJson(windowEnd),
        'label': label.wire,
        'probabilities': probabilities,
        'confidence': confidence,
        'features': features,
        'model': model.toJson(),
      };
}

/// `WellnessSlot` — one time slot of the Module 2 forecast (all scores 0..1).
class WellnessSlot {
  const WellnessSlot({
    required this.start,
    required this.end,
    required this.mentalFatigue,
    required this.focus,
    required this.energy,
    required this.burnoutRisk,
  });

  factory WellnessSlot.fromJson(JsonMap json) => WellnessSlot(
        start: parseDate(json['start']),
        end: parseDate(json['end']),
        mentalFatigue: toDouble(json['mental_fatigue']),
        focus: toDouble(json['focus']),
        energy: toDouble(json['energy']),
        burnoutRisk: toDouble(json['burnout_risk']),
      );

  final DateTime start;
  final DateTime end;
  final double mentalFatigue;
  final double focus;
  final double energy;
  final double burnoutRisk;

  JsonMap toJson() => {
        'start': dateToJson(start),
        'end': dateToJson(end),
        'mental_fatigue': mentalFatigue,
        'focus': focus,
        'energy': energy,
        'burnout_risk': burnoutRisk,
      };
}

/// `WellnessForecast` (Module 2 output).
class WellnessForecast {
  const WellnessForecast({
    required this.userId,
    required this.generatedAt,
    required this.slotMinutes,
    required this.slots,
    required this.model,
    this.sleepHoursLastNight,
  });

  factory WellnessForecast.fromJson(JsonMap json) => WellnessForecast(
        userId: json['user_id'] as String,
        generatedAt: parseDate(json['generated_at']),
        slotMinutes: toInt(json['slot_minutes']),
        slots: parseList(json['slots'], WellnessSlot.fromJson),
        sleepHoursLastNight: toDoubleOrNull(json['sleep_hours_last_night']),
        model: ModelInfo.fromJson(json['model'] as JsonMap),
      );

  final String userId;
  final DateTime generatedAt;
  final int slotMinutes;
  final List<WellnessSlot> slots;
  final double? sleepHoursLastNight;
  final ModelInfo model;

  /// Slots that have not ended yet, earliest first.
  List<WellnessSlot> upcoming(DateTime now, {int count = 6}) {
    final list = slots.where((s) => s.end.isAfter(now)).toList()..sort((a, b) => a.start.compareTo(b.start));
    return list.take(count).toList();
  }

  JsonMap toJson() => {
        'user_id': userId,
        'generated_at': dateToJson(generatedAt),
        'slot_minutes': slotMinutes,
        'slots': slots.map((e) => e.toJson()).toList(),
        'sleep_hours_last_night': sleepHoursLastNight,
        'model': model.toJson(),
      };
}

/// `GET /v1/insights/latest` → `{"affective": ..., "wellness": ...}` (either may be null).
class Insights {
  const Insights({this.affective, this.wellness});

  factory Insights.fromJson(JsonMap json) => Insights(
        affective: json['affective'] == null
            ? null
            : AffectiveStateResult.fromJson(json['affective'] as JsonMap),
        wellness: json['wellness'] == null ? null : WellnessForecast.fromJson(json['wellness'] as JsonMap),
      );

  final AffectiveStateResult? affective;
  final WellnessForecast? wellness;
}
