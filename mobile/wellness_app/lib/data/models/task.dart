import '../../core/utils/json.dart';
import 'wire_enums.dart';

/// `Task` contract (`contracts/schemas/task.schema.json`).
class Task {
  const Task({
    required this.id,
    required this.title,
    required this.category,
    required this.estimatedMinutes,
    required this.priority,
    required this.status,
    required this.createdAt,
    this.deadline,
    this.fixedStart,
    this.fixedEnd,
    this.notes,
    this.completedAt,
  });

  factory Task.fromJson(JsonMap json) => Task(
        id: json['id'] as String,
        title: json['title'] as String,
        category: TaskCategory.fromWire(json['category']),
        estimatedMinutes: toInt(json['estimated_minutes']),
        priority: toIntOrNull(json['priority']) ?? 3,
        status: TaskStatus.fromWire(json['status']),
        createdAt: parseDate(json['created_at']),
        deadline: parseDateOrNull(json['deadline']),
        fixedStart: parseDateOrNull(json['fixed_start']),
        fixedEnd: parseDateOrNull(json['fixed_end']),
        notes: json['notes'] as String?,
        completedAt: parseDateOrNull(json['completed_at']),
      );

  final String id;
  final String title;
  final TaskCategory category;
  final int estimatedMinutes;
  final int priority;
  final TaskStatus status;
  final DateTime createdAt;
  final DateTime? deadline;
  final DateTime? fixedStart;
  final DateTime? fixedEnd;
  final String? notes;
  final DateTime? completedAt;

  bool get isFixed => fixedStart != null && fixedEnd != null;
  bool get isOpen => status == TaskStatus.pending || status == TaskStatus.scheduled;

  JsonMap toJson() => {
        'id': id,
        'title': title,
        'category': category.wire,
        'estimated_minutes': estimatedMinutes,
        'priority': priority,
        'status': status.wire,
        'created_at': dateToJson(createdAt),
        'deadline': dateToJsonOrNull(deadline),
        'fixed_start': dateToJsonOrNull(fixedStart),
        'fixed_end': dateToJsonOrNull(fixedEnd),
        'notes': notes,
        'completed_at': dateToJsonOrNull(completedAt),
      };
}

/// `TaskCreate` contract (`POST /v1/tasks`).
class TaskCreate {
  const TaskCreate({
    required this.title,
    required this.estimatedMinutes,
    this.category = TaskCategory.other,
    this.priority = 3,
    this.deadline,
    this.fixedStart,
    this.fixedEnd,
    this.notes,
  });

  final String title;
  final TaskCategory category;
  final int estimatedMinutes;
  final int priority;
  final DateTime? deadline;
  final DateTime? fixedStart;
  final DateTime? fixedEnd;
  final String? notes;

  JsonMap toJson() => {
        'title': title,
        'category': category.wire,
        'estimated_minutes': estimatedMinutes,
        'priority': priority,
        'deadline': dateToJsonOrNull(deadline),
        'fixed_start': dateToJsonOrNull(fixedStart),
        'fixed_end': dateToJsonOrNull(fixedEnd),
        'notes': notes,
      };
}

/// `TaskUpdate` contract (`PATCH /v1/tasks/{id}`). Only non-null fields are sent.
/// Note: the contract does not allow changing a fixed time — delete and re-create the task instead.
class TaskUpdate {
  const TaskUpdate({
    this.title,
    this.category,
    this.estimatedMinutes,
    this.deadline,
    this.priority,
    this.status,
    this.notes,
  });

  final String? title;
  final TaskCategory? category;
  final int? estimatedMinutes;
  final DateTime? deadline;
  final int? priority;
  final TaskStatus? status;
  final String? notes;

  JsonMap toJson() => {
        if (title != null) 'title': title,
        if (category != null) 'category': category!.wire,
        if (estimatedMinutes != null) 'estimated_minutes': estimatedMinutes,
        if (deadline != null) 'deadline': dateToJson(deadline!),
        if (priority != null) 'priority': priority,
        if (status != null) 'status': status!.wire,
        if (notes != null) 'notes': notes,
      };
}
