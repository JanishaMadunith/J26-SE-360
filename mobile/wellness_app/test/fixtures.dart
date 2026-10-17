// Sample objects shared by the tests. Timestamps are UTC with milliseconds so that
// fromJson → toJson reproduces the exact same strings.
import 'package:wellness_app/data/models/insights.dart';
import 'package:wellness_app/data/models/model_info.dart';
import 'package:wellness_app/data/models/recommendations.dart';
import 'package:wellness_app/data/models/schedule.dart';
import 'package:wellness_app/data/models/sensing.dart';
import 'package:wellness_app/data/models/wire_enums.dart';

final t0 = DateTime.utc(2026, 10, 17, 8);

DateTime at(int minutes) => t0.add(Duration(minutes: minutes));

SensorBatch sampleBatch() => SensorBatch(
      batchId: '7d0c2f0e-2b8f-4a4e-9b7a-2f8a6b1d9c01',
      deviceId: 'pixel-7a-test',
      appVersion: '0.1.0',
      timezone: 'Asia/Colombo',
      collectedFrom: t0,
      collectedTo: at(30),
      ppg: [
        PPGRecording(startedAt: at(1), fps: 30, tMs: const [0, 33, 66], red: const [200, 201.5, 199], green: const [50, 51, 49], blue: const [20, 20, 21]),
      ],
      keystrokes: [KeystrokeEvent(downAt: at(2), holdMs: 0, keyClass: KeyClass.letter)],
      screenEvents: [ScreenEvent(at: at(3), type: ScreenEventType.unlock)],
      appUsage: [AppUsageSession(start: at(4), end: at(9), category: AppCategory.social, appHash: '0123456789abcdef')],
      activity: [ActivityTransition(at: at(10), activity: ActivityType.walking, entering: true)],
      steps: [StepCount(start: t0, end: at(30), steps: 412)],
      light: [LightReading(at: at(11), lux: 120.5)],
      selfReports: [SelfReport(at: at(12), mood: 4, stress: 2, energy: 3, focus: 4, note: 'fine')],
    );

Map<String, dynamic> sampleTaskJson() => {
      'id': '5b0e1c7a-9f6c-4e47-8d55-0c3b8b0b6a11',
      'title': 'Write literature review',
      'category': 'study',
      'estimated_minutes': 90,
      'priority': 4,
      'status': 'scheduled',
      'created_at': '2026-10-17T08:00:00.000Z',
      'deadline': '2026-10-20T18:30:00.000Z',
      'fixed_start': null,
      'fixed_end': null,
      'notes': 'Chapter 2',
      'completed_at': null,
    };

const _model = ModelInfo(name: 'baseline', version: '0.1.0');

ScheduleResult sampleSchedule() => ScheduleResult(
      userId: 'u-1',
      generatedAt: t0,
      horizonStart: t0,
      horizonEnd: t0.add(const Duration(days: 7)),
      items: [ScheduledItem(taskId: 'task-1', start: at(60), end: at(150), completionProbability: 0.82)],
      unscheduled: const [UnscheduledTask(taskId: 'task-2', reason: 'no_free_slot_before_deadline')],
      atRiskTaskIds: const ['task-3'],
      overloadPredicted: false,
      reason: 'scheduled',
      model: _model,
    );

AffectiveStateResult sampleAffective() => AffectiveStateResult(
      userId: 'u-1',
      windowStart: t0,
      windowEnd: at(60),
      label: AffectiveLabel.relaxed,
      probabilities: const {'stress': 0.1, 'relaxed': 0.6, 'happy': 0.2, 'sad': 0.05, 'anxiety': 0.05},
      confidence: 0.6,
      features: const {'hr_bpm': 72.0, 'hrv_rmssd_ms': null},
      model: _model,
    );

WellnessForecast sampleForecast() => WellnessForecast(
      userId: 'u-1',
      generatedAt: t0,
      slotMinutes: 60,
      slots: [
        WellnessSlot(start: t0, end: at(60), mentalFatigue: 0.3, focus: 0.7, energy: 0.65, burnoutRisk: 0.1),
      ],
      sleepHoursLastNight: 7.2,
      model: _model,
    );

Recommendations sampleRecommendations() => Recommendations(
      userId: 'u-1',
      generatedAt: t0,
      meals: [
        MealRecommendation(
          mealType: MealType.lunch,
          windowStart: at(240),
          windowEnd: at(285),
          foods: const [FoodItem(name: 'Red rice & dhal', portion: '1 plate', calories: 550, reason: 'Steady energy')],
          rationale: 'Long study block this afternoon',
        ),
      ],
      suggestedActivities: const [
        SuggestedActivity(title: '10-minute walk', category: TaskCategory.exercise, estimatedMinutes: 10, reason: 'Stress is high'),
      ],
      model: _model,
    );
