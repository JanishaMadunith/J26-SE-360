import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/providers.dart';
import '../../core/theme/app_theme.dart';
import '../../core/utils/formatters.dart';
import '../../core/widgets/async_view.dart';
import '../../data/models/schedule.dart';
import '../../data/models/task.dart';
import '../home_dashboard/dashboard_providers.dart';

/// Module 3 output: timeline for today / tomorrow with completion-probability chips.
class ScheduleScreen extends ConsumerStatefulWidget {
  const ScheduleScreen({super.key});

  @override
  ConsumerState<ScheduleScreen> createState() => _ScheduleScreenState();
}

class _ScheduleScreenState extends ConsumerState<ScheduleScreen> {
  int _dayOffset = 0;
  bool _regenerating = false;

  Future<void> _regenerate() async {
    setState(() => _regenerating = true);
    try {
      await ref.read(scheduleApiProvider).regenerate();
      refreshAll(ref);
      if (mounted) showSnack(context, 'Schedule regenerated');
    } catch (e) {
      if (mounted) showSnack(context, '$e');
    } finally {
      if (mounted) setState(() => _regenerating = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final schedule = ref.watch(currentScheduleProvider);
    final tasks = ref.watch(taskIndexProvider).valueOrNull ?? const <String, Task>{};
    final now = DateTime.now();
    final day = DateTime(now.year, now.month, now.day).add(Duration(days: _dayOffset));

    return Scaffold(
      appBar: AppBar(
        title: const Text('Schedule'),
        actions: [
          IconButton(
            tooltip: 'Regenerate',
            onPressed: _regenerating ? null : _regenerate,
            icon: _regenerating
                ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2))
                : const Icon(Icons.refresh),
          ),
        ],
      ),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(12),
            child: SegmentedButton<int>(
              segments: const [
                ButtonSegment(value: 0, label: Text('Today')),
                ButtonSegment(value: 1, label: Text('Tomorrow')),
              ],
              selected: {_dayOffset},
              onSelectionChanged: (s) => setState(() => _dayOffset = s.first),
            ),
          ),
          Expanded(
            child: RefreshIndicator(
              onRefresh: () => ref.refresh(currentScheduleProvider.future),
              child: AsyncView<ScheduleResult?>(
                value: schedule,
                onRetry: () => ref.invalidate(currentScheduleProvider),
                data: (s) => _Timeline(schedule: s, day: day, tasks: tasks),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _Timeline extends StatelessWidget {
  const _Timeline({required this.schedule, required this.day, required this.tasks});

  final ScheduleResult? schedule;
  final DateTime day;
  final Map<String, Task> tasks;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final text = Theme.of(context).textTheme;
    final s = schedule;
    if (s == null) {
      return ListView(
        children: const [
          Padding(
            padding: EdgeInsets.all(32),
            child: Center(child: Text('No schedule yet. Add tasks, then tap refresh to generate one.')),
          ),
        ],
      );
    }
    final items = s.itemsOn(day);
    return ListView(
      padding: const EdgeInsets.fromLTRB(16, 0, 16, 24),
      children: [
        Text('Generated ${formatAgo(s.generatedAt)} · ${s.model} · reason: ${s.reason}', style: text.bodySmall),
        const SizedBox(height: 8),
        if (items.isEmpty) const Padding(padding: EdgeInsets.all(24), child: Center(child: Text('Nothing planned.'))),
        for (final item in items)
          Card(
            child: ListTile(
              leading: SizedBox(
                width: 56,
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [Text(formatTime(item.start), style: text.titleSmall), Text(formatTime(item.end), style: text.bodySmall)],
                ),
              ),
              title: Text(tasks[item.taskId]?.title ?? 'Task'),
              subtitle: Text(item.isFixed ? 'Fixed event' : (tasks[item.taskId]?.category.label ?? '')),
              trailing: Chip(
                label: Text(formatPercent(item.completionProbability)),
                avatar: Icon(Icons.check, size: 16, color: AppTheme.scoreColor(scheme, item.completionProbability)),
              ),
            ),
          ),
        if (s.unscheduled.isNotEmpty) ...[
          const SizedBox(height: 16),
          Text("Couldn't fit", style: text.titleSmall?.copyWith(color: scheme.error)),
          for (final u in s.unscheduled)
            ListTile(
              dense: true,
              leading: Icon(Icons.event_busy, color: scheme.error),
              title: Text(tasks[u.taskId]?.title ?? 'Task'),
              subtitle: Text(u.reason.replaceAll('_', ' ')),
            ),
        ],
      ],
    );
  }
}
