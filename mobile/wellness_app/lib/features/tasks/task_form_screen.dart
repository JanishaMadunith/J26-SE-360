import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../core/providers.dart';
import '../../core/utils/formatters.dart';
import '../../core/widgets/async_view.dart';
import '../../data/models/task.dart';
import '../../data/models/wire_enums.dart';
import '../home_dashboard/dashboard_providers.dart';

/// Add or edit a task. Editing cannot change a fixed time (the TaskUpdate contract has no such field).
class TaskFormScreen extends ConsumerStatefulWidget {
  const TaskFormScreen({this.existing, super.key});

  final Task? existing;

  @override
  ConsumerState<TaskFormScreen> createState() => _TaskFormScreenState();
}

class _TaskFormScreenState extends ConsumerState<TaskFormScreen> {
  final _form = GlobalKey<FormState>();
  late final TextEditingController _title;
  late final TextEditingController _notes;
  late TaskCategory _category;
  late int _minutes;
  late int _priority;
  DateTime? _deadline;
  bool _fixed = false;
  DateTime? _fixedStart;
  bool _saving = false;

  bool get _editing => widget.existing != null;

  @override
  void initState() {
    super.initState();
    final t = widget.existing;
    _title = TextEditingController(text: t?.title ?? '');
    _notes = TextEditingController(text: t?.notes ?? '');
    _category = t?.category ?? TaskCategory.study;
    _minutes = t?.estimatedMinutes ?? 60;
    _priority = t?.priority ?? 3;
    _deadline = t?.deadline;
    _fixed = t?.isFixed ?? false;
    _fixedStart = t?.fixedStart;
  }

  @override
  void dispose() {
    _title.dispose();
    _notes.dispose();
    super.dispose();
  }

  Future<DateTime?> _pickDateTime(DateTime? initial) async {
    final now = DateTime.now();
    final date = await showDatePicker(
      context: context,
      initialDate: initial ?? now,
      firstDate: now.subtract(const Duration(days: 1)),
      lastDate: now.add(const Duration(days: 365)),
    );
    if (date == null || !mounted) return null;
    final time = await showTimePicker(
      context: context,
      initialTime: TimeOfDay.fromDateTime(initial ?? now.add(const Duration(hours: 1))),
    );
    if (time == null) return null;
    return DateTime(date.year, date.month, date.day, time.hour, time.minute);
  }

  Future<void> _save() async {
    if (!_form.currentState!.validate()) return;
    if (_fixed && _fixedStart == null && !_editing) {
      showSnack(context, 'Pick a start time for the fixed event');
      return;
    }
    setState(() => _saving = true);
    final api = ref.read(tasksApiProvider);
    final notes = _notes.text.trim().isEmpty ? null : _notes.text.trim();
    try {
      if (_editing) {
        await api.update(
          widget.existing!.id,
          TaskUpdate(
            title: _title.text.trim(),
            category: _category,
            estimatedMinutes: _minutes,
            priority: _priority,
            deadline: _deadline,
            notes: notes,
          ),
        );
      } else {
        await api.create(
          TaskCreate(
            title: _title.text.trim(),
            category: _category,
            estimatedMinutes: _minutes,
            priority: _priority,
            deadline: _deadline,
            fixedStart: _fixed ? _fixedStart : null,
            fixedEnd: _fixed ? _fixedStart!.add(Duration(minutes: _minutes)) : null,
            notes: notes,
          ),
        );
      }
      refreshAll(ref);
      if (mounted) context.pop();
    } catch (e) {
      if (mounted) showSnack(context, '$e');
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    return Scaffold(
      appBar: AppBar(
        title: Text(_editing ? 'Edit task' : 'New task'),
        actions: [
          TextButton(onPressed: _saving ? null : _save, child: const Text('Save')),
        ],
      ),
      body: Form(
        key: _form,
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            TextFormField(
              controller: _title,
              textCapitalization: TextCapitalization.sentences,
              decoration: const InputDecoration(labelText: 'What do you need to do?'),
              validator: (v) => (v == null || v.trim().isEmpty) ? 'Enter a title' : null,
            ),
            const SizedBox(height: 16),
            Text('Category', style: text.titleSmall),
            Wrap(
              spacing: 8,
              children: [
                for (final c in TaskCategory.values)
                  ChoiceChip(label: Text(c.label), selected: _category == c, onSelected: (_) => setState(() => _category = c)),
              ],
            ),
            const SizedBox(height: 16),
            Text('Estimated time: ${formatMinutes(_minutes)}', style: text.titleSmall),
            Slider(
              value: _minutes.toDouble(),
              min: 5,
              max: 240,
              divisions: 47,
              label: formatMinutes(_minutes),
              onChanged: (v) => setState(() => _minutes = v.round()),
            ),
            Text('Priority', style: text.titleSmall),
            const SizedBox(height: 8),
            SegmentedButton<int>(
              segments: [for (var p = 1; p <= 5; p++) ButtonSegment(value: p, label: Text('$p'))],
              selected: {_priority},
              onSelectionChanged: (s) => setState(() => _priority = s.first),
            ),
            const SizedBox(height: 16),
            ListTile(
              contentPadding: EdgeInsets.zero,
              leading: const Icon(Icons.flag_outlined),
              title: Text(_deadline == null ? 'No deadline' : 'Due ${formatDayTime(_deadline!)}'),
              trailing: _deadline == null
                  ? null
                  : IconButton(icon: const Icon(Icons.clear), onPressed: () => setState(() => _deadline = null)),
              onTap: () async {
                final picked = await _pickDateTime(_deadline);
                if (picked != null) setState(() => _deadline = picked);
              },
            ),
            SwitchListTile(
              contentPadding: EdgeInsets.zero,
              title: const Text('Fixed time (e.g. a lecture)'),
              subtitle: _editing ? const Text('Fixed times cannot be changed after creation') : null,
              value: _fixed,
              onChanged: _editing ? null : (v) => setState(() => _fixed = v),
            ),
            if (_fixed)
              ListTile(
                contentPadding: EdgeInsets.zero,
                leading: const Icon(Icons.schedule),
                title: Text(
                  _fixedStart == null
                      ? 'Pick start time'
                      : '${formatDayTime(_fixedStart!)} (${formatMinutes(_minutes)})',
                ),
                onTap: _editing
                    ? null
                    : () async {
                        final picked = await _pickDateTime(_fixedStart);
                        if (picked != null) setState(() => _fixedStart = picked);
                      },
              ),
            const SizedBox(height: 8),
            TextFormField(
              controller: _notes,
              maxLines: 3,
              maxLength: 1000,
              decoration: const InputDecoration(labelText: 'Notes (optional)'),
            ),
          ],
        ),
      ),
    );
  }
}
