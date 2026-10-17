import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/config/app_config.dart';
import '../../core/providers.dart';
import '../../core/utils/formatters.dart';
import '../../core/widgets/async_view.dart';
import '../../sensing/sensing_providers.dart';
import '../auth/auth_controller.dart';
import 'permissions_service.dart';

final permissionsServiceProvider = Provider((ref) => PermissionsService(ref.watch(nativeBridgeProvider)));

final permissionsStatusProvider =
    FutureProvider.autoDispose<SensingPermissions>((ref) => ref.watch(permissionsServiceProvider).status());

final uploadStatusProvider = FutureProvider.autoDispose<(DateTime?, int)>((ref) async {
  final repo = await ref.watch(sensingRepositoryProvider.future);
  return (repo.lastUploadAt, await repo.pendingUploads());
});

/// Server URL, permissions, sync status. Reachable before login so the server URL can be fixed first.
class SettingsScreen extends ConsumerStatefulWidget {
  const SettingsScreen({super.key});

  @override
  ConsumerState<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends ConsumerState<SettingsScreen> with WidgetsBindingObserver {
  late final TextEditingController _url;
  bool _syncing = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _url = TextEditingController(text: ref.read(appConfigProvider).serverUrl);
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _url.dispose();
    super.dispose();
  }

  /// Returning from system settings: re-check the permissions.
  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed) ref.invalidate(permissionsStatusProvider);
  }

  Future<void> _saveUrl() async {
    await ref.read(appConfigProvider.notifier).setServerUrl(_url.text);
    try {
      final health = await ref.read(healthApiProvider).health();
      if (mounted) showSnack(context, 'Saved. Server status: ${health['status']}');
    } catch (e) {
      if (mounted) showSnack(context, 'Saved, but the server did not answer: $e');
    }
  }

  Future<void> _syncNow() async {
    setState(() => _syncing = true);
    try {
      final summary = await syncNow(ref);
      if (mounted) showSnack(context, summary);
    } catch (e) {
      if (mounted) showSnack(context, 'Sync failed: $e');
    } finally {
      ref.invalidate(uploadStatusProvider);
      if (mounted) setState(() => _syncing = false);
    }
  }

  Widget _permissionTile({
    required String title,
    required String subtitle,
    required bool granted,
    required Future<void> Function() onGrant,
  }) =>
      ListTile(
        leading: Icon(granted ? Icons.check_circle : Icons.error_outline,
            color: granted ? Theme.of(context).colorScheme.primary : Theme.of(context).colorScheme.error),
        title: Text(title),
        subtitle: Text(subtitle),
        trailing: granted
            ? null
            : TextButton(
                onPressed: () async {
                  await onGrant();
                  ref.invalidate(permissionsStatusProvider);
                },
                child: const Text('Grant'),
              ),
      );

  @override
  Widget build(BuildContext context) {
    final perms = ref.watch(permissionsStatusProvider);
    final upload = ref.watch(uploadStatusProvider);
    final loggedIn = ref.watch(authControllerProvider).valueOrNull?.isLoggedIn ?? false;
    final service = ref.read(permissionsServiceProvider);
    final text = Theme.of(context).textTheme;

    return Scaffold(
      appBar: AppBar(title: const Text('Settings')),
      body: ListView(
        padding: const EdgeInsets.symmetric(vertical: 8),
        children: [
          Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('Server', style: text.titleMedium),
                const SizedBox(height: 8),
                TextField(
                  controller: _url,
                  keyboardType: TextInputType.url,
                  decoration: InputDecoration(
                    labelText: 'Core API URL',
                    helperText: 'USB: ${AppConfig.defaultServerUrl} with "adb reverse tcp:8000 tcp:8000", '
                        'or your computer\'s LAN IP, e.g. http://192.168.1.20:8000',
                    helperMaxLines: 3,
                  ),
                ),
                const SizedBox(height: 8),
                Align(
                  alignment: Alignment.centerRight,
                  child: FilledButton.tonal(onPressed: _saveUrl, child: const Text('Save & test')),
                ),
              ],
            ),
          ),
          const Divider(),
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 8, 16, 0),
            child: Text('Sensing permissions', style: text.titleMedium),
          ),
          AsyncView<SensingPermissions>(
            value: perms,
            onRetry: () => ref.invalidate(permissionsStatusProvider),
            data: (p) => Column(
              children: [
                _permissionTile(
                  title: 'Usage access',
                  subtitle: 'Screen and app-category usage (Module 1). Opens system settings.',
                  granted: p.usageAccess,
                  onGrant: service.requestUsageAccess,
                ),
                _permissionTile(
                  title: 'Typing rhythm service',
                  subtitle: 'Accessibility service that records key timing only — never what you type.',
                  granted: p.typingService,
                  onGrant: service.requestTypingService,
                ),
                _permissionTile(
                  title: 'Physical activity',
                  subtitle: 'Steps and activity type.',
                  granted: p.activity,
                  onGrant: service.requestActivity,
                ),
                _permissionTile(
                  title: 'Notifications',
                  subtitle: 'Check-in reminders and schedule changes.',
                  granted: p.notifications,
                  onGrant: service.requestNotifications,
                ),
                _permissionTile(
                  title: 'Camera',
                  subtitle: 'Heart-rate measurement with your fingertip.',
                  granted: p.camera,
                  onGrant: service.requestCamera,
                ),
              ],
            ),
          ),
          if (loggedIn) ...[
            const Divider(),
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 8, 16, 0),
              child: Text('Data upload', style: text.titleMedium),
            ),
            AsyncView<(DateTime?, int)>(
              value: upload,
              onRetry: () => ref.invalidate(uploadStatusProvider),
              data: (u) => ListTile(
                leading: const Icon(Icons.cloud_upload_outlined),
                title: Text('Last upload: ${formatAgo(u.$1)}'),
                subtitle: Text('${u.$2} batch(es) waiting · background sync every 15 minutes'),
                trailing: _syncing
                    ? const SizedBox(width: 24, height: 24, child: CircularProgressIndicator(strokeWidth: 2))
                    : TextButton(onPressed: _syncNow, child: const Text('Sync now')),
              ),
            ),
            const Divider(),
            ListTile(
              leading: const Icon(Icons.logout),
              title: const Text('Log out'),
              onTap: () => unawaited(ref.read(authControllerProvider.notifier).logout()),
            ),
          ],
          Padding(
            padding: const EdgeInsets.all(16),
            child: Text('App ${AppConfig.appVersion} · consent ${AppConfig.consentVersion}', style: text.bodySmall),
          ),
        ],
      ),
    );
  }
}
