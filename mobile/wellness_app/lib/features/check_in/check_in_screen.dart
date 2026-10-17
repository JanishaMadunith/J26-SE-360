import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../core/providers.dart';
import '../../core/widgets/async_view.dart';
import '../../data/models/sensing.dart';
import '../../sensing/sensing_providers.dart';

/// Self-report check-in (1–5). These are the ground-truth labels Modules 1 and 2 train on.
class CheckInScreen extends ConsumerStatefulWidget {
  const CheckInScreen({super.key});

  @override
  ConsumerState<CheckInScreen> createState() => _CheckInScreenState();
}

class _CheckInScreenState extends ConsumerState<CheckInScreen> {
  final Map<String, int> _values = {'mood': 3, 'stress': 3, 'energy': 3, 'focus': 3};
  final _note = TextEditingController();
  bool _saving = false;

  static const _labels = {
    'mood': ('Mood', 'Very low', 'Very good'),
    'stress': ('Stress', 'Calm', 'Very stressed'),
    'energy': ('Energy', 'Exhausted', 'Full of energy'),
    'focus': ('Focus', 'Scattered', 'Very focused'),
  };

  @override
  void dispose() {
    _note.dispose();
    super.dispose();
  }

  Future<void> _submit() async {
    setState(() => _saving = true);
    final report = SelfReport(
      at: DateTime.now(),
      mood: _values['mood'],
      stress: _values['stress'],
      energy: _values['energy'],
      focus: _values['focus'],
      note: _note.text.trim().isEmpty ? null : _note.text.trim(),
    );
    var message = 'Thanks — check-in saved';
    try {
      await ref.read(sensingApiProvider).postSelfReport(report);
    } catch (_) {
      // Offline: keep it in the upload queue inside a batch instead.
      try {
        final repo = await ref.read(sensingRepositoryProvider.future);
        await repo.enqueue(repo.singleBatch(selfReports: [report]));
        message = 'Saved offline — it will upload when you are back online';
      } catch (e) {
        message = 'Could not save the check-in: $e';
      }
    }
    if (!mounted) return;
    setState(() => _saving = false);
    showSnack(context, message);
    context.pop();
  }

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    return Scaffold(
      appBar: AppBar(title: const Text('How are you right now?')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          for (final key in _values.keys) ...[
            Text(_labels[key]!.$1, style: text.titleMedium),
            Slider(
              value: _values[key]!.toDouble(),
              min: 1,
              max: 5,
              divisions: 4,
              label: '${_values[key]}',
              onChanged: (v) => setState(() => _values[key] = v.round()),
            ),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [Text(_labels[key]!.$2, style: text.bodySmall), Text(_labels[key]!.$3, style: text.bodySmall)],
            ),
            const SizedBox(height: 16),
          ],
          TextField(
            controller: _note,
            maxLength: 280,
            decoration: const InputDecoration(labelText: 'Anything to add? (optional)'),
          ),
          const SizedBox(height: 16),
          FilledButton(onPressed: _saving ? null : _submit, child: const Text('Save check-in')),
        ],
      ),
    );
  }
}
