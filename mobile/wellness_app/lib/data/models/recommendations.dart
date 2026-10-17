import '../../core/utils/json.dart';
import 'model_info.dart';
import 'wire_enums.dart';

class FoodItem {
  const FoodItem({required this.name, required this.portion, required this.reason, this.calories});

  factory FoodItem.fromJson(JsonMap json) => FoodItem(
        name: json['name'] as String,
        portion: json['portion'] as String,
        calories: toIntOrNull(json['calories']),
        reason: json['reason'] as String,
      );

  final String name;
  final String portion;
  final int? calories;
  final String reason;

  JsonMap toJson() => {'name': name, 'portion': portion, 'calories': calories, 'reason': reason};
}

class MealRecommendation {
  const MealRecommendation({
    required this.mealType,
    required this.windowStart,
    required this.windowEnd,
    required this.foods,
    required this.rationale,
  });

  factory MealRecommendation.fromJson(JsonMap json) => MealRecommendation(
        mealType: MealType.fromWire(json['meal_type']),
        windowStart: parseDate(json['window_start']),
        windowEnd: parseDate(json['window_end']),
        foods: parseList(json['foods'], FoodItem.fromJson),
        rationale: json['rationale'] as String,
      );

  final MealType mealType;
  final DateTime windowStart;
  final DateTime windowEnd;
  final List<FoodItem> foods;
  final String rationale;

  JsonMap toJson() => {
        'meal_type': mealType.wire,
        'window_start': dateToJson(windowStart),
        'window_end': dateToJson(windowEnd),
        'foods': foods.map((e) => e.toJson()).toList(),
        'rationale': rationale,
      };
}

/// Module 4 "Part B". Shown only; if accepted, the app creates a normal task that Module 3 then schedules.
class SuggestedActivity {
  const SuggestedActivity({
    required this.title,
    required this.category,
    required this.estimatedMinutes,
    required this.reason,
  });

  factory SuggestedActivity.fromJson(JsonMap json) => SuggestedActivity(
        title: json['title'] as String,
        category: TaskCategory.fromWire(json['category']),
        estimatedMinutes: toInt(json['estimated_minutes']),
        reason: json['reason'] as String,
      );

  final String title;
  final TaskCategory category;
  final int estimatedMinutes;
  final String reason;

  JsonMap toJson() => {
        'title': title,
        'category': category.wire,
        'estimated_minutes': estimatedMinutes,
        'reason': reason,
      };
}

/// `Recommendations` contract (`GET /v1/recommendations/latest`).
class Recommendations {
  const Recommendations({
    required this.userId,
    required this.generatedAt,
    required this.meals,
    required this.model,
    this.suggestedActivities = const [],
  });

  factory Recommendations.fromJson(JsonMap json) => Recommendations(
        userId: json['user_id'] as String,
        generatedAt: parseDate(json['generated_at']),
        meals: parseList(json['meals'], MealRecommendation.fromJson),
        suggestedActivities: parseList(json['suggested_activities'], SuggestedActivity.fromJson),
        model: ModelInfo.fromJson(json['model'] as JsonMap),
      );

  final String userId;
  final DateTime generatedAt;
  final List<MealRecommendation> meals;
  final List<SuggestedActivity> suggestedActivities;
  final ModelInfo model;

  JsonMap toJson() => {
        'user_id': userId,
        'generated_at': dateToJson(generatedAt),
        'meals': meals.map((e) => e.toJson()).toList(),
        'suggested_activities': suggestedActivities.map((e) => e.toJson()).toList(),
        'model': model.toJson(),
      };
}
