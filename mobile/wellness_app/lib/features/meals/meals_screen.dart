import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/providers.dart';
import '../../core/utils/formatters.dart';
import '../../core/widgets/async_view.dart';
import '../../data/models/recommendations.dart';
import '../../data/models/task.dart';
import '../home_dashboard/dashboard_providers.dart';

/// Module 4 output: meal recommendations + suggested activities ("add as task" hands them to Module 3).
class MealsScreen extends ConsumerWidget {
  const MealsScreen({super.key});

  Future<void> _addAsTask(BuildContext context, WidgetRef ref, SuggestedActivity a) async {
    try {
      await ref.read(tasksApiProvider).create(
            TaskCreate(title: a.title, category: a.category, estimatedMinutes: a.estimatedMinutes),
          );
      refreshAll(ref);
      if (context.mounted) showSnack(context, 'Added "${a.title}" — it will be scheduled on the next run');
    } catch (e) {
      if (context.mounted) showSnack(context, '$e');
    }
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final recs = ref.watch(recommendationsProvider);
    final text = Theme.of(context).textTheme;
    return Scaffold(
      appBar: AppBar(title: const Text('Meals')),
      body: RefreshIndicator(
        onRefresh: () => ref.refresh(recommendationsProvider.future),
        child: AsyncView<Recommendations?>(
          value: recs,
          onRetry: () => ref.invalidate(recommendationsProvider),
          data: (r) {
            if (r == null) {
              return ListView(
                children: const [
                  Padding(padding: EdgeInsets.all(32), child: Center(child: Text('No recommendations yet.'))),
                ],
              );
            }
            return ListView(
              padding: const EdgeInsets.all(16),
              children: [
                Text('Updated ${formatAgo(r.generatedAt)} · ${r.model}', style: text.bodySmall),
                const SizedBox(height: 8),
                for (final meal in r.meals)
                  Card(
                    child: Padding(
                      padding: const EdgeInsets.all(16),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            '${meal.mealType.label} · ${formatTimeRange(meal.windowStart, meal.windowEnd)}',
                            style: text.titleMedium,
                          ),
                          const SizedBox(height: 4),
                          Text(meal.rationale, style: text.bodySmall),
                          const Divider(),
                          for (final f in meal.foods)
                            ListTile(
                              dense: true,
                              contentPadding: EdgeInsets.zero,
                              title: Text('${f.name} — ${f.portion}'),
                              subtitle: Text(f.reason),
                              trailing: f.calories == null ? null : Text('${f.calories} kcal'),
                            ),
                        ],
                      ),
                    ),
                  ),
                if (r.suggestedActivities.isNotEmpty) ...[
                  const SizedBox(height: 16),
                  Text('Suggested for you', style: text.titleMedium),
                  for (final a in r.suggestedActivities)
                    Card(
                      child: ListTile(
                        title: Text(a.title),
                        subtitle: Text('${formatMinutes(a.estimatedMinutes)} · ${a.reason}'),
                        trailing: TextButton(
                          onPressed: () => _addAsTask(context, ref, a),
                          child: const Text('Add as task'),
                        ),
                      ),
                    ),
                ],
              ],
            );
          },
        ),
      ),
    );
  }
}
