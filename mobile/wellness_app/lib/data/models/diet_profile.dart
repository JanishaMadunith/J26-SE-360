import '../../core/utils/json.dart';

/// `DietProfile` contract (Module 4 input, `GET/PUT /v1/me/diet-profile`).
class DietProfile {
  const DietProfile({
    this.vegetarian = false,
    this.vegan = false,
    this.halal = false,
    this.allergies = const [],
    this.dislikes = const [],
    this.healthConditions = const [],
    this.dailyCalorieTarget,
  });

  factory DietProfile.fromJson(JsonMap json) => DietProfile(
        vegetarian: (json['vegetarian'] as bool?) ?? false,
        vegan: (json['vegan'] as bool?) ?? false,
        halal: (json['halal'] as bool?) ?? false,
        allergies: parseStrings(json['allergies']),
        dislikes: parseStrings(json['dislikes']),
        healthConditions: parseStrings(json['health_conditions']),
        dailyCalorieTarget: toIntOrNull(json['daily_calorie_target']),
      );

  final bool vegetarian;
  final bool vegan;
  final bool halal;
  final List<String> allergies;
  final List<String> dislikes;
  final List<String> healthConditions;
  final int? dailyCalorieTarget;

  JsonMap toJson() => {
        'vegetarian': vegetarian,
        'vegan': vegan,
        'halal': halal,
        'allergies': allergies,
        'dislikes': dislikes,
        'health_conditions': healthConditions,
        'daily_calorie_target': dailyCalorieTarget,
      };

  DietProfile copyWith({
    bool? vegetarian,
    bool? vegan,
    bool? halal,
    List<String>? allergies,
    List<String>? dislikes,
    List<String>? healthConditions,
    int? dailyCalorieTarget,
  }) =>
      DietProfile(
        vegetarian: vegetarian ?? this.vegetarian,
        vegan: vegan ?? this.vegan,
        halal: halal ?? this.halal,
        allergies: allergies ?? this.allergies,
        dislikes: dislikes ?? this.dislikes,
        healthConditions: healthConditions ?? this.healthConditions,
        dailyCalorieTarget: dailyCalorieTarget ?? this.dailyCalorieTarget,
      );
}
