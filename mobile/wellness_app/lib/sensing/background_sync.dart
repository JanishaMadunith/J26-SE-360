import 'dart:ui';

import 'package:shared_preferences/shared_preferences.dart';
import 'package:workmanager/workmanager.dart';

import 'sensing_repository.dart';

const kPeriodicSyncTask = 'lk.sliit.j26se360.periodicSync';

/// Entry point of the workmanager background isolate. Must be a top-level function.
@pragma('vm:entry-point')
void callbackDispatcher() {
  Workmanager().executeTask((task, inputData) async {
    try {
      DartPluginRegistrant.ensureInitialized();
      final prefs = await SharedPreferences.getInstance();
      await prefs.reload(); // values written by the UI isolate since this isolate started
      // No NativeBridge here: the MethodChannel lives only in MainActivity's engine.
      // Usage/screen/typing data arrive through the JSONL files written by Kotlin.
      final repo = await SensingRepository.standalone(prefs);
      await repo.collectAndEnqueue();
      await repo.flush();
    } catch (_) {
      // Swallow: the offline queue keeps data until the next run. Returning false would only make
      // WorkManager retry immediately with its own back-off.
    }
    return true;
  });
}

class BackgroundSync {
  /// Call once from `main()`. Re-registering an existing periodic task keeps the existing schedule.
  static Future<void> initialize() async {
    await Workmanager().initialize(callbackDispatcher);
    await Workmanager().registerPeriodicTask(
      kPeriodicSyncTask,
      kPeriodicSyncTask,
      frequency: const Duration(minutes: 15), // Android's minimum period
    );
  }
}
