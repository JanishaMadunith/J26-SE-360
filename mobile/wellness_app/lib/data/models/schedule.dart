import '../../core/utils/json.dart';
import 'model_info.dart';

/// `ScheduledItem` (Module 3 output).
class ScheduledItem {
  const ScheduledItem({
    required this.taskId,
    required this.start,
    required this.end,
    required this.completionProbability,
    this.isFixed = false,
  });

  factory ScheduledItem.fromJson(JsonMap json) => ScheduledItem(
        taskId: json['task_id'] as String,
        start: parseDate(json['start']),
        end: parseDate(json['end']),
        completionProbability: toDouble(json['completion_probability']),
        isFixed: (json['is_fixed'] as bool?) ?? false,
      );

  final String taskId;
  final DateTime start;
  final DateTime end;
  final double completionProbability;
  final bool isFixed;

  JsonMap toJson() => {
        'task_id': taskId,
        'start': dateToJson(start),
        'end': dateToJson(end),
        'completion_probability': completionProbability,
        'is_fixed': isFixed,
      };
}

class UnscheduledTask {
  const UnscheduledTask({required this.taskId, required this.reason});

  factory UnscheduledTask.fromJson(JsonMap json) =>
      UnscheduledTask(taskId: json['task_id'] as String, reason: json['reason'] as String);

  final String taskId;
  final String reason;

  JsonMap toJson() => {'task_id': taskId, 'reason': reason};
}

/// `ScheduleResult` contract (`GET /v1/schedule/current`, `POST /v1/schedule/regenerate`).
class ScheduleResult {
  const ScheduleResult({
    required this.userId,
    required this.generatedAt,
    required this.horizonStart,
    required this.horizonEnd,
    required this.items,
    required this.reason,
    required this.model,
    this.unscheduled = const [],
    this.atRiskTaskIds = const [],
    this.overloadPredicted = false,
  });

  factory ScheduleResult.fromJson(JsonMap json) => ScheduleResult(
        userId: json['user_id'] as String,
        generatedAt: parseDate(json['generated_at']),
        horizonStart: parseDate(json['horizon_start']),
        horizonEnd: parseDate(json['horizon_end']),
        items: parseList(json['items'], ScheduledItem.fromJson),
        unscheduled: parseList(json['unscheduled'], UnscheduledTask.fromJson),
        atRiskTaskIds: parseStrings(json['at_risk_task_ids']),
        overloadPredicted: (json['overload_predicted'] as bool?) ?? false,
        reason: json['reason'] as String,
        model: ModelInfo.fromJson(json['model'] as JsonMap),
      );

  final String userId;
  final DateTime generatedAt;
  final DateTime horizonStart;
  final DateTime horizonEnd;
  final List<ScheduledItem> items;
  final List<UnscheduledTask> unscheduled;
  final List<String> atRiskTaskIds;
  final bool overloadPredicted;
  final String reason;
  final ModelInfo model;

  /// Items whose start falls on the given local calendar day, sorted by start time.
  List<ScheduledItem> itemsOn(DateTime day) {
    final list = items
        .where((i) => i.start.year == day.year && i.start.month == day.month && i.start.day == day.day)
        .toList()
      ..sort((a, b) => a.start.compareTo(b.start));
    return list;
  }

  JsonMap toJson() => {
        'user_id': userId,
        'generated_at': dateToJson(generatedAt),
        'horizon_start': dateToJson(horizonStart),
        'horizon_end': dateToJson(horizonEnd),
        'items': items.map((e) => e.toJson()).toList(),
        'unscheduled': unscheduled.map((e) => e.toJson()).toList(),
        'at_risk_task_ids': atRiskTaskIds,
        'overload_predicted': overloadPredicted,
        'reason': reason,
        'model': model.toJson(),
      };
}
