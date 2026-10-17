import 'dart:convert';
import 'dart:io';

import 'package:path/path.dart' as p;

import '../data/models/sensing.dart';
import 'native_bridge.dart';

/// Reads and writes the JSON-lines buffers shared with the Kotlin side (`filesDir/sensing/*.jsonl`).
///
/// Files are the hand-off point because the workmanager background isolate cannot call the MethodChannel
/// registered by MainActivity. Draining is "rename, then read, then delete", so a writer appending at the
/// same moment simply starts a new file and no line is ever read twice.
class FileBridge {
  FileBridge(this._dirResolver);

  static const keystrokes = 'keystrokes.jsonl';
  static const appUsage = 'app_usage.jsonl';
  static const screen = 'screen.jsonl';
  static const activity = 'activity.jsonl';

  final Future<Directory> Function() _dirResolver;

  Future<void> append(String name, Map<String, dynamic> record) async {
    final dir = await _dirResolver();
    if (!dir.existsSync()) dir.createSync(recursive: true);
    await File(p.join(dir.path, name)).writeAsString('${jsonEncode(record)}\n', mode: FileMode.append, flush: true);
  }

  /// Returns every buffered record for [name] and removes them. Also recovers files left over from an
  /// interrupted earlier drain (`<name>.draining-*`).
  Future<List<Map<String, dynamic>>> drain(String name) async {
    final dir = await _dirResolver();
    if (!dir.existsSync()) return const [];
    final out = <Map<String, dynamic>>[];
    final current = File(p.join(dir.path, name));
    if (current.existsSync()) {
      final claimed = p.join(dir.path, '$name.draining-${DateTime.now().microsecondsSinceEpoch}');
      try {
        current.renameSync(claimed);
      } on FileSystemException {
        // Another drain claimed it first — fine.
      }
    }
    final pending = dir
        .listSync()
        .whereType<File>()
        .where((f) => p.basename(f.path).startsWith('$name.draining-'))
        .toList()
      ..sort((a, b) => a.path.compareTo(b.path));
    for (final f in pending) {
      for (final line in await f.readAsLines()) {
        if (line.trim().isEmpty) continue;
        try {
          final decoded = jsonDecode(line);
          if (decoded is Map<String, dynamic>) out.add(decoded);
        } on FormatException {
          // Skip a partially written line rather than losing the whole file.
        }
      }
      await f.delete();
    }
    return out;
  }

  Future<List<KeystrokeEvent>> drainKeystrokes() async => (await drain(keystrokes)).map(keystrokeFromNative).toList();

  Future<List<AppUsageSession>> drainAppUsage() async => (await drain(appUsage)).map(appUsageFromNative).toList();

  Future<List<ScreenEvent>> drainScreenEvents() async => (await drain(screen)).map(screenEventFromNative).toList();

  Future<List<ActivityTransition>> drainActivity() async => (await drain(activity)).map(activityFromNative).toList();
}
