import 'dart:async';
import 'dart:io';

import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:uuid/uuid.dart';

import '../core/config/app_config.dart';
import '../core/network/api_client.dart';
import '../core/storage/token_store.dart';
import '../data/api/apis.dart';
import '../data/models/sensing.dart';
import 'batch_builder.dart';
import 'collectors/light_collector.dart';
import 'collectors/steps_collector.dart';
import 'file_bridge.dart';
import 'native_bridge.dart';
import 'sqflite_queue_store.dart';
import 'upload_queue.dart';

/// Collects everything since the last collection into a [SensorBatch], stores it in the offline queue,
/// and uploads queued batches. Used by the UI isolate and by the 15-minute background worker.
class SensingRepository {
  SensingRepository({
    required this.prefs,
    required this.queue,
    required this.api,
    required this.files,
    NativeBridge? native,
    StepsCollector? steps,
    LightCollector? light,
  })  : _native = native,
        _steps = steps ?? StepsCollector(),
        _light = light ?? LightCollector();

  final SharedPreferences prefs;
  final UploadQueue queue;
  final SensingApi api;
  final FileBridge files;
  final NativeBridge? _native;
  final StepsCollector _steps;
  final LightCollector _light;

  static const _uuid = Uuid();

  /// Builds a repository without Riverpod (used in the background isolate).
  static Future<SensingRepository> standalone(SharedPreferences prefs, {NativeBridge? native}) async {
    final dio = createDio(baseUrl: AppConfig.load(prefs).serverUrl, tokens: TokenStore());
    return SensingRepository(
      prefs: prefs,
      queue: UploadQueue(await SqfliteQueueStore.open()),
      api: SensingApi(dio),
      files: FileBridge(() => sensingDir(prefs)),
      native: native,
    );
  }

  /// Directory shared with Kotlin (`filesDir/sensing`). The exact path is cached from the native side
  /// the first time the app runs; `getApplicationSupportDirectory()` maps to `filesDir` on Android.
  static Future<Directory> sensingDir(SharedPreferences prefs) async {
    final cached = prefs.getString(AppConfig.kSensingDir);
    if (cached != null) return Directory(cached);
    return Directory(p.join((await getApplicationSupportDirectory()).path, 'sensing'));
  }

  String deviceId() {
    var id = prefs.getString(AppConfig.kDeviceId);
    if (id == null) {
      id = _uuid.v4();
      unawaited(prefs.setString(AppConfig.kDeviceId, id));
    }
    return id;
  }

  String timezone() => prefs.getString(AppConfig.kTimezone) ?? AppConfig.defaultTimezone;

  DateTime? get lastUploadAt {
    final ms = prefs.getInt(AppConfig.kLastUploadAt);
    return ms == null ? null : DateTime.fromMillisecondsSinceEpoch(ms);
  }

  /// UI isolate only: cache native facts the background isolate cannot ask for, and ask Kotlin to write a
  /// fresh usage/screen snapshot so this collection includes the last few minutes.
  Future<void> refreshNativeContext() async {
    final native = _native;
    if (native == null) return;
    try {
      final dir = await native.getSensingDir();
      if (dir != null) await prefs.setString(AppConfig.kSensingDir, dir);
      final tz = await native.getTimezone();
      if (tz != null) await prefs.setString(AppConfig.kTimezone, tz);
      await native.snapshotNow();
    } catch (_) {
      // Native side unavailable (e.g. tests) — background snapshots still happen natively.
    }
  }

  /// Drains every buffer and builds one batch. Drained data only lives in memory until
  /// [collectAndEnqueue] stores it, so always use that from app code.
  Future<SensorBatch> collect({DateTime? now}) async {
    final to = now ?? DateTime.now();
    final lastMs = prefs.getInt(AppConfig.kLastCollectedAt);
    final from = lastMs == null ? to.subtract(const Duration(hours: 1)) : DateTime.fromMillisecondsSinceEpoch(lastMs);

    final keystrokes = await files.drainKeystrokes();
    final appUsage = await files.drainAppUsage();
    final screen = await files.drainScreenEvents();
    final activity = await files.drainActivity();
    final step = await _steps.readSince(prefs, to);
    final light = await _light.sample();

    final batch = BatchBuilder.build(
      batchId: _uuid.v4(),
      deviceId: deviceId(),
      appVersion: AppConfig.appVersion,
      timezone: timezone(),
      from: from.isBefore(to) ? from : to,
      to: to,
      keystrokes: keystrokes,
      appUsage: appUsage,
      screenEvents: screen,
      activity: activity,
      steps: [if (step != null) step],
      light: light,
    );
    await prefs.setInt(AppConfig.kLastCollectedAt, to.millisecondsSinceEpoch);
    return batch;
  }

  /// Collect + persist in one step. Returns the batch (possibly empty — empty batches are not queued).
  Future<SensorBatch> collectAndEnqueue({DateTime? now}) async {
    final batch = await collect(now: now);
    if (!batch.isEmpty) await queue.enqueue(batch);
    return batch;
  }

  /// Queue a batch built elsewhere (PPG measurement, offline check-in).
  Future<void> enqueue(SensorBatch batch) => queue.enqueue(batch);

  /// Builds a batch holding only the given items (stamped with this device's metadata).
  SensorBatch singleBatch({List<PPGRecording> ppg = const [], List<SelfReport> selfReports = const []}) {
    final now = DateTime.now();
    return BatchBuilder.build(
      batchId: _uuid.v4(),
      deviceId: deviceId(),
      appVersion: AppConfig.appVersion,
      timezone: timezone(),
      from: now,
      to: now,
      ppg: ppg,
      selfReports: selfReports,
    );
  }

  Future<FlushResult> flush() async {
    final result = await queue.flush(api.uploadBatchJson);
    if (result.sent > 0) await prefs.setInt(AppConfig.kLastUploadAt, DateTime.now().millisecondsSinceEpoch);
    return result;
  }

  Future<int> pendingUploads() => queue.pending();
}
