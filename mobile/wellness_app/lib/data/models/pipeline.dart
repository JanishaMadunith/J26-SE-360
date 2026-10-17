import '../../core/utils/json.dart';

class StepStatus {
  const StepStatus({
    required this.step,
    required this.ok,
    required this.ms,
    this.usedFallback = false,
    this.error,
  });

  factory StepStatus.fromJson(JsonMap json) => StepStatus(
        step: json['step'] as String,
        ok: json['ok'] as bool,
        ms: toInt(json['ms']),
        usedFallback: (json['used_fallback'] as bool?) ?? false,
        error: json['error'] as String?,
      );

  /// One of `affective`, `wellness`, `schedule`, `recommendations`.
  final String step;
  final bool ok;
  final int ms;
  final bool usedFallback;
  final String? error;
}

/// `POST /v1/pipeline/run` response.
class PipelineRun {
  const PipelineRun({
    required this.id,
    required this.status,
    required this.steps,
    required this.startedAt,
    this.finishedAt,
  });

  factory PipelineRun.fromJson(JsonMap json) => PipelineRun(
        id: '${json['id']}',
        status: json['status'] as String,
        steps: parseList(json['steps'], StepStatus.fromJson),
        startedAt: parseDate(json['started_at']),
        finishedAt: parseDateOrNull(json['finished_at']),
      );

  final String id;

  /// `ok`, `partial` or `failed`.
  final String status;
  final List<StepStatus> steps;
  final DateTime startedAt;
  final DateTime? finishedAt;
}
