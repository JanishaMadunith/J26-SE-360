// Checks every Dart model against the JSON Schemas exported from the Python contracts
// (contracts/schemas/*.schema.json). If a teammate changes a contract and runs `make contracts`,
// this test fails until the Dart model is updated — so the app and the backend cannot drift apart.
import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:wellness_app/data/models/diet_profile.dart';
import 'package:wellness_app/data/models/sensing.dart';
import 'package:wellness_app/data/models/task.dart';
import 'package:wellness_app/data/models/wire_enums.dart';

import 'fixtures.dart';

const _schemaDir = '../../contracts/schemas';

Map<String, dynamic> _load(String name) =>
    jsonDecode(File('$_schemaDir/$name.schema.json').readAsStringSync()) as Map<String, dynamic>;

Map<String, dynamic> _def(Map<String, dynamic> root, String name) =>
    (root[r'$defs'] as Map<String, dynamic>)[name] as Map<String, dynamic>;

/// Every key we send must exist in the schema, and every required key must be sent.
void _expectObject(Map<String, dynamic> json, Map<String, dynamic> schema, String what) {
  final props = (schema['properties'] as Map<String, dynamic>).keys.toSet();
  final required = ((schema['required'] as List<dynamic>?) ?? const <dynamic>[]).map((e) => e as String).toSet();
  final keys = json.keys.toSet();
  expect(keys.difference(props), isEmpty, reason: '$what: keys not in schema');
  expect(required.difference(keys), isEmpty, reason: '$what: required keys missing');
}

void _expectEnum(Map<String, dynamic> root, String defName, List<WireEnum> values) {
  final schemaValues = (_def(root, defName)['enum'] as List<dynamic>).map((e) => e as String).toSet();
  expect(values.map((v) => v.wire).toSet(), schemaValues, reason: 'enum $defName');
}

Map<String, dynamic> _roundTrip(Map<String, dynamic> json) => jsonDecode(jsonEncode(json)) as Map<String, dynamic>;

void main() {
  final missing = !Directory(_schemaDir).existsSync();
  final skip = missing ? 'contracts/schemas not found (run tests from mobile/wellness_app)' : null;

  group('contracts', () {
    test('SensorBatch and every nested record match sensor_batch.schema.json', () {
      final root = _load('sensor_batch');
      final batch = _roundTrip(sampleBatch().toJson());
      _expectObject(batch, root, 'SensorBatch');
      final nested = {
        'ppg': 'PPGRecording',
        'keystrokes': 'KeystrokeEvent',
        'screen_events': 'ScreenEvent',
        'app_usage': 'AppUsageSession',
        'activity': 'ActivityTransition',
        'steps': 'StepCount',
        'light': 'LightReading',
        'self_reports': 'SelfReport',
      };
      nested.forEach((field, def) {
        final list = batch[field] as List<dynamic>;
        expect(list, isNotEmpty, reason: 'fixture should exercise $field');
        for (final item in list) {
          _expectObject(item as Map<String, dynamic>, _def(root, def), def);
        }
      });
      _expectEnum(root, 'KeyClass', KeyClass.values);
      _expectEnum(root, 'ScreenEventType', ScreenEventType.values);
      _expectEnum(root, 'AppCategory', AppCategory.values);
      _expectEnum(root, 'ActivityType', ActivityType.values);
    }, skip: skip);

    test('SensorBatchAck', () {
      _expectObject(
        const SensorBatchAck(batchId: 'b', accepted: true, counts: {'ppg': 1}).toJson(),
        _load('sensor_batch_ack'),
        'SensorBatchAck',
      );
    }, skip: skip);

    test('Task, TaskCreate, TaskUpdate', () {
      final task = Task.fromJson(sampleTaskJson());
      final taskSchema = _load('task');
      _expectObject(_roundTrip(task.toJson()), taskSchema, 'Task');
      _expectEnum(taskSchema, 'TaskCategory', TaskCategory.values);
      _expectEnum(taskSchema, 'TaskStatus', TaskStatus.values);
      _expectObject(
        _roundTrip(TaskCreate(title: 'Read', estimatedMinutes: 30, deadline: DateTime.utc(2026, 11)).toJson()),
        _load('task_create'),
        'TaskCreate',
      );
      _expectObject(
        _roundTrip(const TaskUpdate(title: 'x', priority: 2, status: TaskStatus.done).toJson()),
        _load('task_update'),
        'TaskUpdate',
      );
    }, skip: skip);

    test('ScheduleResult (Module 3)', () {
      final root = _load('m3_schedule_result');
      final json = _roundTrip(sampleSchedule().toJson());
      _expectObject(json, root, 'ScheduleResult');
      _expectObject((json['items'] as List<dynamic>).first as Map<String, dynamic>, _def(root, 'ScheduledItem'), 'ScheduledItem');
      _expectObject(
        (json['unscheduled'] as List<dynamic>).first as Map<String, dynamic>,
        _def(root, 'UnscheduledTask'),
        'UnscheduledTask',
      );
    }, skip: skip);

    test('AffectiveStateResult (Module 1) and WellnessForecast (Module 2)', () {
      final m1 = _load('m1_affective_state_result');
      _expectObject(_roundTrip(sampleAffective().toJson()), m1, 'AffectiveStateResult');
      _expectEnum(m1, 'AffectiveLabel', AffectiveLabel.values);
      final m2 = _load('m2_wellness_forecast');
      final forecast = _roundTrip(sampleForecast().toJson());
      _expectObject(forecast, m2, 'WellnessForecast');
      _expectObject((forecast['slots'] as List<dynamic>).first as Map<String, dynamic>, _def(m2, 'WellnessSlot'), 'WellnessSlot');
    }, skip: skip);

    test('Recommendations (Module 4) and DietProfile', () {
      final root = _load('m4_recommendations');
      final json = _roundTrip(sampleRecommendations().toJson());
      _expectObject(json, root, 'Recommendations');
      final meal = (json['meals'] as List<dynamic>).first as Map<String, dynamic>;
      _expectObject(meal, _def(root, 'MealRecommendation'), 'MealRecommendation');
      _expectObject((meal['foods'] as List<dynamic>).first as Map<String, dynamic>, _def(root, 'FoodItem'), 'FoodItem');
      _expectObject(
        (json['suggested_activities'] as List<dynamic>).first as Map<String, dynamic>,
        _def(root, 'SuggestedActivity'),
        'SuggestedActivity',
      );
      _expectEnum(root, 'MealType', MealType.values);
      final request = _load('m4_recommendation_request');
      _expectObject(_roundTrip(const DietProfile(allergies: ['peanut']).toJson()), _def(request, 'DietProfile'), 'DietProfile');
    }, skip: skip);
  });
}
