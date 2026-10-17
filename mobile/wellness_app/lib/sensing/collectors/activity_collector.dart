import 'dart:async';

import 'package:flutter_activity_recognition/flutter_activity_recognition.dart' as ar;

import '../../data/models/wire_enums.dart';
import '../file_bridge.dart';

/// Listens to activity-recognition updates while the app process is alive and records *transitions*
/// (exit old activity, enter new one) into the shared `activity.jsonl` buffer.
class ActivityCollector {
  ActivityCollector(this._files);

  final FileBridge _files;
  StreamSubscription<ar.Activity>? _sub;
  ActivityType? _current;

  bool get isRunning => _sub != null;

  Future<bool> start() async {
    if (_sub != null) return true;
    final recognition = ar.FlutterActivityRecognition.instance;
    var permission = await recognition.checkPermission();
    if (permission != ar.ActivityPermission.GRANTED) {
      permission = await recognition.requestPermission();
    }
    if (permission != ar.ActivityPermission.GRANTED) return false;
    _sub = recognition.activityStream.listen(_onActivity, onError: (Object _) {});
    return true;
  }

  Future<void> stop() async {
    await _sub?.cancel();
    _sub = null;
  }

  Future<void> _onActivity(ar.Activity activity) async {
    final type = mapActivityName(activity.type.name);
    if (type == _current) return;
    final now = DateTime.now().millisecondsSinceEpoch;
    if (_current != null) {
      await _files.append(FileBridge.activity, {'at_ms': now, 'activity': _current!.wire, 'entering': false});
    }
    await _files.append(FileBridge.activity, {'at_ms': now, 'activity': type.wire, 'entering': true});
    _current = type;
  }

  /// Maps the plugin's enum name (e.g. `IN_VEHICLE`, `ON_BICYCLE`) to the contract enum.
  /// Matching on the name keeps this working if the plugin renames its enum casing.
  static ActivityType mapActivityName(String name) {
    final n = name.toLowerCase();
    if (n.contains('vehicle')) return ActivityType.inVehicle;
    if (n.contains('bicycle')) return ActivityType.onBicycle;
    if (n.contains('running')) return ActivityType.running;
    if (n.contains('walking') || n.contains('on_foot')) return ActivityType.walking;
    if (n.contains('still')) return ActivityType.still;
    return ActivityType.unknown;
  }
}
