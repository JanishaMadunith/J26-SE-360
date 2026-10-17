import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:wellness_app/data/models/auth.dart';
import 'package:wellness_app/data/models/insights.dart';
import 'package:wellness_app/data/models/recommendations.dart';
import 'package:wellness_app/data/models/schedule.dart';
import 'package:wellness_app/data/models/sensing.dart';
import 'package:wellness_app/data/models/task.dart';
import 'package:wellness_app/data/models/user.dart';

import 'fixtures.dart';

/// fromJson(toJson(x)) must give back identical JSON (snake_case keys, UTC timestamps).
void expectStable(Map<String, dynamic> Function() toJson, Map<String, dynamic> Function(Map<String, dynamic>) roundTrip) {
  final first = jsonDecode(jsonEncode(toJson())) as Map<String, dynamic>;
  final second = jsonDecode(jsonEncode(roundTrip(first))) as Map<String, dynamic>;
  expect(second, equals(first));
}

void main() {
  test('Task round-trips the backend JSON exactly', () {
    final json = sampleTaskJson();
    expect(Task.fromJson(json).toJson(), equals(json));
  });

  test('TaskUpdate only sends fields that were set', () {
    expect(const TaskUpdate(priority: 5).toJson(), {'priority': 5});
  });

  test('SensorBatch', () {
    expectStable(() => sampleBatch().toJson(), (j) => SensorBatch.fromJson(j).toJson());
    final batch = sampleBatch();
    expect(batch.isEmpty, isFalse);
    expect(batch.counts['ppg'], 1);
    expect(batch.toJson()['collected_from'], '2026-10-17T08:00:00.000Z');
  });

  test('ScheduleResult', () {
    expectStable(() => sampleSchedule().toJson(), (j) => ScheduleResult.fromJson(j).toJson());
    final s = sampleSchedule();
    expect(s.itemsOn(s.items.first.start.toLocal()), hasLength(1));
  });

  test('Insights', () {
    expectStable(() => sampleAffective().toJson(), (j) => AffectiveStateResult.fromJson(j).toJson());
    expectStable(() => sampleForecast().toJson(), (j) => WellnessForecast.fromJson(j).toJson());
    final insights = Insights.fromJson({'affective': null, 'wellness': sampleForecast().toJson()});
    expect(insights.affective, isNull);
    expect(insights.wellness!.slots, hasLength(1));
  });

  test('Recommendations', () {
    expectStable(() => sampleRecommendations().toJson(), (j) => Recommendations.fromJson(j).toJson());
  });

  test('TokenResponse with embedded user', () {
    final json = {
      'access_token': 'a',
      'refresh_token': 'r',
      'token_type': 'bearer',
      'user': {
        'id': 'u-1',
        'email': 'a@b.lk',
        'display_name': 'Dilsara',
        'timezone': 'Asia/Colombo',
        'created_at': '2026-10-17T08:00:00.000Z',
        'consent_version': null,
        'consented_at': null,
      },
    };
    final r = TokenResponse.fromJson(json);
    expect(r.user!.displayName, 'Dilsara');
    expect(User.fromJson(json['user']! as Map<String, dynamic>).toJson(), json['user']);
    expect(r.toJson(), json);
  });
}
