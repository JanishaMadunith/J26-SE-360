import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../core/providers.dart';
import '../../core/router/app_router.dart';
import '../../core/utils/formatters.dart';
import '../../core/widgets/async_view.dart';
import '../../data/models/task.dart';
import '../../data/models/wire_enums.dart';
import '../home_dashboard/dashboard_providers.dart';

class TasksScreen extends ConsumerWidget {
  const TasksScreen({super.key});

  Future<void> _complete(BuildContext context, WidgetRef ref, Task task) async {
    try {
      await ref.read(tasksApiProvider).complete(task.id);
      refreshAll(ref);
    } catch (e) {
      if (context.mounted) showSnack(context, '$e');
    }
  }

  Future<void> _delete(BuildContext context, WidgetRef ref, Task task) async {
    try {
      await ref.read(tasksApiProvider).delete(task.id);
      refreshAll(ref);
      if (context.mounted) showSnack(context, 'Deleted "${task.title}"');
    } catch (e) {
      if (context.mounted) showSnack(context, '$e');
    }
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final tasks = ref.watch(tasksProvider);
    return Scaffold(
      appBar: AppBar(title: const Text('Tasks')),
      floatingActionButton: FloatingActionButton(
        tooltip: 'Add task',
        onPressed: () => context.push(Routes.newTask),
        child: const Icon(Icons.add),
      ),
      body: RefreshIndicator(
        onRefresh: () => ref.refresh(tasksProvider.future),
        child: AsyncView<List<Task>>(
          value: tasks,
          onRetry: () => ref.invalidate(tasksProvider),
          data: (all) {
            final open = all.where((t) => t.isOpen).toList()
              ..sort((a, b) => (a.deadline ?? DateTime(9999)).compareTo(b.deadline ?? DateTime(9999)));
            final closed = all.where((t) => !t.isOpen && t.status != TaskStatus.cancelled).toList()
              ..sort((a, b) => (b.completedAt ?? b.createdAt).compareTo(a.completedAt ?? a.createdAt));
            if (all.isEmpty) {
              return ListView(
                children: const [
                  Padding(padding: EdgeInsets.all(32), child: Center(child: Text('No tasks yet. Tap + to add one.'))),
                ],
              );
            }
            return ListView(
              padding: const EdgeInsets.only(bottom: 88),
              children: [
                _Header('To do (${open.length})'),
                for (final t in open)
                  _TaskTile(
                    task: t,
                    onComplete: () => _complete(context, ref, t),
                    onDelete: () => _delete(context, ref, t),
                  ),
                if (closed.isNotEmpty) _Header('Finished (${closed.length})'),
                for (final t in closed.take(30)) _TaskTile(task: t),
              ],
            );
          },
        ),
      ),
    );
  }
}

class _Header extends StatelessWidget {
  const _Header(this.text);

  final String text;

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.fromLTRB(16, 16, 16, 4),
        child: Text(text, style: Theme.of(context).textTheme.titleSmall),
      );
}

class _TaskTile extends StatelessWidget {
  const _TaskTile({required this.task, this.onComplete, this.onDelete});

  final Task task;
  final VoidCallback? onComplete;
  final VoidCallback? onDelete;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final details = <String>[
      task.category.label,
      formatMinutes(task.estimatedMinutes),
      'P${task.priority}',
      if (task.isFixed) 'fixed ${formatDayTime(task.fixedStart!)}',
      if (task.deadline != null) 'due ${formatDayTime(task.deadline!)}',
    ];
    final overdue = task.isOpen && task.deadline != null && task.deadline!.isBefore(DateTime.now());
    return ListTile(
      leading: task.isOpen
          ? IconButton(
              tooltip: 'Mark done',
              icon: const Icon(Icons.radio_button_unchecked),
              onPressed: onComplete,
            )
          : Icon(
              task.status == TaskStatus.done ? Icons.check_circle : Icons.cancel_outlined,
              color: task.status == TaskStatus.done ? scheme.primary : scheme.error,
            ),
      title: Text(task.title),
      subtitle: Text(details.join(' · '), style: overdue ? TextStyle(color: scheme.error) : null),
      trailing: task.isOpen
          ? PopupMenuButton<String>(
              onSelected: (v) {
                if (v == 'edit') unawaited(context.push(Routes.editTask, extra: task));
                if (v == 'delete') onDelete?.call();
              },
              itemBuilder: (_) => const [
                PopupMenuItem(value: 'edit', child: Text('Edit')),
                PopupMenuItem(value: 'delete', child: Text('Delete')),
              ],
            )
          : Text(task.status.label),
      onTap: task.isOpen ? () => context.push(Routes.editTask, extra: task) : null,
    );
  }
}
