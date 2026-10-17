import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/providers.dart';
import '../../data/models/insights.dart';
import '../../data/models/recommendations.dart';
import '../../data/models/schedule.dart';
import '../../data/models/task.dart';

/// Shared read models. Screens call `ref.invalidate(...)` after a change to refresh them.
final insightsProvider = FutureProvider.autoDispose<Insights>((ref) => ref.watch(insightsApiProvider).latest());

final currentScheduleProvider =
    FutureProvider.autoDispose<ScheduleResult?>((ref) => ref.watch(scheduleApiProvider).current());

final tasksProvider = FutureProvider.autoDispose<List<Task>>((ref) => ref.watch(tasksApiProvider).list());

final recommendationsProvider =
    FutureProvider.autoDispose<Recommendations?>((ref) => ref.watch(recommendationsApiProvider).latest());

/// Task lookup by id (schedule items only carry the task id).
final taskIndexProvider = FutureProvider.autoDispose<Map<String, Task>>((ref) async {
  final tasks = await ref.watch(tasksProvider.future);
  return {for (final t in tasks) t.id: t};
});

void refreshAll(WidgetRef ref) {
  ref
    ..invalidate(insightsProvider)
    ..invalidate(currentScheduleProvider)
    ..invalidate(tasksProvider)
    ..invalidate(recommendationsProvider);
}
