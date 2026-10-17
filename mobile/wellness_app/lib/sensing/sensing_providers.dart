import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../core/providers.dart';
import 'collectors/activity_collector.dart';
import 'file_bridge.dart';
import 'native_bridge.dart';
import 'sensing_repository.dart';
import 'sqflite_queue_store.dart';
import 'upload_queue.dart';

final nativeBridgeProvider = Provider<NativeBridge>((ref) => NativeBridge());

final fileBridgeProvider = Provider<FileBridge>((ref) {
  final prefs = ref.watch(sharedPreferencesProvider);
  return FileBridge(() => SensingRepository.sensingDir(prefs));
});

final sensingRepositoryProvider = FutureProvider<SensingRepository>((ref) async {
  final store = await SqfliteQueueStore.open(); // sqflite returns the same instance for the same path
  return SensingRepository(
    prefs: ref.watch(sharedPreferencesProvider),
    queue: UploadQueue(store),
    api: ref.watch(sensingApiProvider),
    files: ref.watch(fileBridgeProvider),
    native: ref.watch(nativeBridgeProvider),
  );
});

final activityCollectorProvider = Provider<ActivityCollector>((ref) {
  final collector = ActivityCollector(ref.watch(fileBridgeProvider));
  ref.onDispose(collector.stop);
  return collector;
});

/// Foreground "sync now": native snapshot → collect → queue → upload. Returns a short summary.
Future<String> syncNow(WidgetRef ref) async {
  final repo = await ref.read(sensingRepositoryProvider.future);
  await repo.refreshNativeContext();
  final batch = await repo.collectAndEnqueue();
  final result = await repo.flush();
  final pending = await repo.pendingUploads();
  final collected = batch.counts.entries.where((e) => e.value > 0).map((e) => '${e.value} ${e.key}').join(', ');
  return 'Collected ${collected.isEmpty ? 'nothing new' : collected}. Uploaded: $result. Waiting: $pending.';
}
