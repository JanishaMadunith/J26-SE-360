import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/config/app_config.dart';
import '../auth/auth_controller.dart';

/// Research consent. Text must match the ethics-approved information sheet — update it (and bump
/// [AppConfig.consentVersion]) once the team's ethics application is approved.
class ConsentScreen extends ConsumerStatefulWidget {
  const ConsentScreen({super.key});

  @override
  ConsumerState<ConsentScreen> createState() => _ConsentScreenState();
}

class _ConsentScreenState extends ConsumerState<ConsentScreen> {
  bool _agree = false;
  bool _busy = false;

  static const _collected = [
    ('Heart rate (camera)', 'Average colour of each camera frame while your finger covers the lens — never photos or video.'),
    ('Typing rhythm', 'When keys are pressed and whether they were letters, digits, spaces or deletes — never what you type. Password fields are ignored.'),
    ('Phone usage', 'Screen on/off/unlock times and which category of app was open (e.g. social, game). App names are replaced by an anonymous code.'),
    ('Movement', 'Activity type (still, walking, …) and step counts.'),
    ('Light', 'Ambient light level, used to estimate sleep.'),
    ('Check-ins', 'Mood, stress, energy and focus ratings you enter.'),
    ('Tasks & meals', 'Tasks you add and your diet preferences, used to plan your day and suggest meals.'),
  ];

  Future<void> _accept() async {
    setState(() => _busy = true);
    try {
      await ref.read(authControllerProvider.notifier).acceptConsent();
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Could not save consent: $e')));
      }
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    return Scaffold(
      appBar: AppBar(title: const Text('Before you start')),
      body: SafeArea(
        child: Column(
          children: [
            Expanded(
              child: ListView(
                padding: const EdgeInsets.all(16),
                children: [
                  Text(
                    'This app is part of the SLIIT research project J26-SE-360. It collects the data below to '
                    'estimate your wellness, plan your schedule and suggest meals. All analysis happens on the '
                    'project server. You can stop at any time by uninstalling the app or asking the team to delete '
                    'your data.',
                    style: text.bodyMedium,
                  ),
                  const SizedBox(height: 16),
                  Text('What is collected', style: text.titleMedium),
                  for (final (title, detail) in _collected)
                    ListTile(
                      contentPadding: EdgeInsets.zero,
                      leading: const Icon(Icons.check_circle_outline),
                      title: Text(title),
                      subtitle: Text(detail),
                    ),
                  const SizedBox(height: 8),
                  Text('Consent version ${AppConfig.consentVersion}', style: text.bodySmall),
                ],
              ),
            ),
            CheckboxListTile(
              value: _agree,
              onChanged: (v) => setState(() => _agree = v ?? false),
              title: const Text('I have read this and agree to take part'),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
              child: SizedBox(
                width: double.infinity,
                child: FilledButton(
                  onPressed: (_agree && !_busy) ? _accept : null,
                  child: const Text('Continue'),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
