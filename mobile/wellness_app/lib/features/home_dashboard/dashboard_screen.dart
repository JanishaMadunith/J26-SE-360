import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../core/providers.dart';
import '../../core/router/app_router.dart';
import '../../core/theme/app_theme.dart';
import '../../core/utils/formatters.dart';
import '../../core/widgets/async_view.dart';
import '../../data/models/insights.dart';
import '../../data/models/schedule.dart';
import '../../data/models/task.dart';
import '../auth/auth_controller.dart';
import 'dashboard_providers.dart';

class DashboardScreen extends ConsumerStatefulWidget {
  const DashboardScreen({super.key});

  @override
  ConsumerState<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends ConsumerState<DashboardScreen> {
  bool _running = false;

  Future<void> _runAnalysis() async {
    setState(() => _running = true);
    try {
      final run = await ref.read(pipelineApiProvider).run();
      if (!mounted) return;
      final failed = run.steps.where((s) => !s.ok).map((s) => s.step).join(', ');
      showSnack(context, failed.isEmpty ? 'Analysis updated (${run.status})' : 'Partly updated — failed: $failed');
      refreshAll(ref);
    } catch (e) {
      if (mounted) showSnack(context, '$e');
    } finally {
      if (mounted) setState(() => _running = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final user = ref.watch(authControllerProvider).valueOrNull?.user;
    final insights = ref.watch(insightsProvider);
    final schedule = ref.watch(currentScheduleProvider);
    final tasks = ref.watch(taskIndexProvider);

    return Scaffold(
      appBar: AppBar(
        title: Text(user == null ? 'Today' : 'Hi, ${user.displayName.split(' ').first}'),
        actions: [
          IconButton(
            tooltip: 'Measure heart rate',
            icon: const Icon(Icons.favorite_outline),
            onPressed: () => context.push(Routes.ppg),
          ),
          IconButton(
            tooltip: 'Check in',
            icon: const Icon(Icons.mood),
            onPressed: () => context.push(Routes.checkIn),
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _running ? null : _runAnalysis,
        icon: _running
            ? const SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2))
            : const Icon(Icons.auto_awesome),
        label: const Text('Analyse now'),
      ),
      body: RefreshIndicator(
        onRefresh: () async {
          refreshAll(ref);
          await ref.read(insightsProvider.future);
        },
        child: ListView(
          padding: const EdgeInsets.fromLTRB(16, 8, 16, 96),
          children: [
            AsyncView<Insights>(
              value: insights,
              onRetry: () => ref.invalidate(insightsProvider),
              data: (i) => Column(
                children: [
                  _AffectiveCard(affective: i.affective),
                  const SizedBox(height: 12),
                  _WellnessCard(forecast: i.wellness),
                ],
              ),
            ),
            const SizedBox(height: 12),
            AsyncView<ScheduleResult?>(
              value: schedule,
              onRetry: () => ref.invalidate(currentScheduleProvider),
              data: (s) => _TodayCard(schedule: s, tasks: tasks.valueOrNull ?? const {}),
            ),
          ],
        ),
      ),
    );
  }
}

class _AffectiveCard extends StatelessWidget {
  const _AffectiveCard({required this.affective});

  final AffectiveStateResult? affective;

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    final a = affective;
    return Card(
      child: ListTile(
        leading: const Icon(Icons.psychology_alt_outlined, size: 36),
        title: Text(a == null ? 'No state detected yet' : 'You seem ${a.label.label.toLowerCase()}', style: text.titleMedium),
        subtitle: Text(
          a == null
              ? 'Keep the app installed — it needs about an hour of data. Try a heart-rate measurement.'
              : 'Confidence ${formatPercent(a.confidence)} · ${formatTimeRange(a.windowStart, a.windowEnd)}'
                  '${a.features['hr_bpm'] != null ? ' · ${a.features['hr_bpm']!.round()} bpm' : ''}',
        ),
      ),
    );
  }
}

class _WellnessCard extends StatelessWidget {
  const _WellnessCard({required this.forecast});

  final WellnessForecast? forecast;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final text = Theme.of(context).textTheme;
    final f = forecast;
    final slots = f?.upcoming(DateTime.now()) ?? const <WellnessSlot>[];
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Energy & focus — next hours', style: text.titleMedium),
            if (f?.sleepHoursLastNight != null)
              Text('Estimated sleep last night: ${f!.sleepHoursLastNight!.toStringAsFixed(1)} h', style: text.bodySmall),
            const SizedBox(height: 12),
            if (slots.isEmpty) const Text('No forecast yet.'),
            for (final s in slots)
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 4),
                child: Row(
                  children: [
                    SizedBox(width: 52, child: Text(formatTime(s.start), style: text.bodySmall)),
                    Expanded(child: _Bar(label: 'Energy', value: s.energy, color: AppTheme.scoreColor(scheme, s.energy))),
                    const SizedBox(width: 8),
                    Expanded(child: _Bar(label: 'Focus', value: s.focus, color: AppTheme.scoreColor(scheme, s.focus))),
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }
}

class _Bar extends StatelessWidget {
  const _Bar({required this.label, required this.value, required this.color});

  final String label;
  final double value;
  final Color color;

  @override
  Widget build(BuildContext context) => Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('$label ${formatPercent(value)}', style: Theme.of(context).textTheme.labelSmall),
          const SizedBox(height: 2),
          LinearProgressIndicator(value: value.clamp(0, 1).toDouble(), color: color, minHeight: 6),
        ],
      );
}

class _TodayCard extends StatelessWidget {
  const _TodayCard({required this.schedule, required this.tasks});

  final ScheduleResult? schedule;
  final Map<String, Task> tasks;

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    final scheme = Theme.of(context).colorScheme;
    final s = schedule;
    final today = s?.itemsOn(DateTime.now()) ?? const <ScheduledItem>[];
    final atRisk = s?.atRiskTaskIds ?? const <String>[];
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(child: Text("Today's plan", style: text.titleMedium)),
                TextButton(onPressed: () => context.go(Routes.schedule), child: const Text('Full schedule')),
              ],
            ),
            if (s?.overloadPredicted ?? false)
              Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: Text('Your day looks overloaded — consider moving something.', style: TextStyle(color: scheme.error)),
              ),
            if (today.isEmpty) const Text('Nothing scheduled today. Add tasks and tap "Analyse now".'),
            for (final item in today)
              ListTile(
                dense: true,
                contentPadding: EdgeInsets.zero,
                leading: Text(formatTime(item.start)),
                title: Text(tasks[item.taskId]?.title ?? 'Task'),
                trailing: Text(
                  formatPercent(item.completionProbability),
                  style: TextStyle(color: AppTheme.scoreColor(scheme, item.completionProbability)),
                ),
              ),
            if (atRisk.isNotEmpty) ...[
              const Divider(),
              Text('At risk of being missed', style: text.titleSmall?.copyWith(color: scheme.error)),
              for (final id in atRisk)
                ListTile(
                  dense: true,
                  contentPadding: EdgeInsets.zero,
                  leading: Icon(Icons.warning_amber_rounded, color: scheme.error),
                  title: Text(tasks[id]?.title ?? 'Task'),
                ),
            ],
          ],
        ),
      ),
    );
  }
}
