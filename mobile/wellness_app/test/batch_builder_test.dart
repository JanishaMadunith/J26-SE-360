import 'package:flutter_test/flutter_test.dart';
import 'package:wellness_app/data/models/sensing.dart';
import 'package:wellness_app/data/models/wire_enums.dart';
import 'package:wellness_app/sensing/batch_builder.dart';
import 'package:wellness_app/sensing/collectors/activity_collector.dart';
import 'package:wellness_app/sensing/native_bridge.dart';

void main() {
  final t0 = DateTime.utc(2026, 10, 17, 8);
  DateTime at(int m) => t0.add(Duration(minutes: m));

  SensorBatch build({
    List<KeystrokeEvent> keystrokes = const [],
    List<AppUsageSession> appUsage = const [],
    List<ScreenEvent> screen = const [],
  }) =>
      BatchBuilder.build(
        batchId: 'b-1',
        deviceId: 'dev-1',
        appVersion: '0.1.0',
        timezone: 'Asia/Colombo',
        from: at(0),
        to: at(15),
        keystrokes: keystrokes,
        appUsage: appUsage,
        screenEvents: screen,
      );

  test('window widens to include events buffered before the last collection', () {
    final b = build(keystrokes: [KeystrokeEvent(downAt: at(-40), holdMs: 0, keyClass: KeyClass.letter)]);
    expect(b.collectedFrom, at(-40));
    expect(b.collectedTo, at(15));
    expect(b.collectedFrom.isAfter(b.collectedTo), isFalse);
  });

  test('events are sorted and zero-length app sessions dropped', () {
    final b = build(
      keystrokes: [
        KeystrokeEvent(downAt: at(5), holdMs: 0, keyClass: KeyClass.space),
        KeystrokeEvent(downAt: at(1), holdMs: 0, keyClass: KeyClass.letter),
      ],
      appUsage: [
        AppUsageSession(start: at(2), end: at(2), category: AppCategory.game, appHash: 'aaaaaaaaaaaaaaaa'),
        AppUsageSession(start: at(3), end: at(8), category: AppCategory.social, appHash: 'bbbbbbbbbbbbbbbb'),
      ],
      screen: [ScreenEvent(at: at(9), type: ScreenEventType.screenOff), ScreenEvent(at: at(4), type: ScreenEventType.unlock)],
    );
    expect(b.keystrokes.map((k) => k.keyClass), [KeyClass.letter, KeyClass.space]);
    expect(b.appUsage, hasLength(1));
    expect(b.screenEvents.first.type, ScreenEventType.unlock);
  });

  test('empty batch is reported as empty', () {
    expect(build().isEmpty, isTrue);
  });

  test('step counter delta handles reboots', () {
    expect(BatchBuilder.stepDelta(previous: 1000, current: 1250), 250);
    expect(BatchBuilder.stepDelta(previous: 9000, current: 120), 120); // counter reset by reboot
  });

  test('native JSON (epoch ms) maps to contract models', () {
    final ms = at(0).millisecondsSinceEpoch;
    final k = keystrokeFromNative({'down_at_ms': ms, 'hold_ms': 0, 'key_class': 'backspace'});
    expect(k.keyClass, KeyClass.backspace);
    expect(k.downAt.toUtc(), at(0));
    final s = screenEventFromNative({'at_ms': ms, 'type': 'screen_on'});
    expect(s.type, ScreenEventType.screenOn);
    final a = appUsageFromNative({'start_ms': ms, 'end_ms': ms + 60000, 'category': 'productivity', 'app_hash': '0123456789abcdef'});
    expect(a.category, AppCategory.productivity);
    final act = activityFromNative({'at_ms': ms, 'activity': 'on_bicycle', 'entering': true});
    expect(act.activity, ActivityType.onBicycle);
  });

  test('activity plugin names map to contract values', () {
    expect(ActivityCollector.mapActivityName('IN_VEHICLE'), ActivityType.inVehicle);
    expect(ActivityCollector.mapActivityName('onBicycle'), ActivityType.onBicycle);
    expect(ActivityCollector.mapActivityName('WALKING'), ActivityType.walking);
    expect(ActivityCollector.mapActivityName('STILL'), ActivityType.still);
    expect(ActivityCollector.mapActivityName('TILTING'), ActivityType.unknown);
  });
}
