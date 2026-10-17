/// Enums whose JSON ("wire") values match `contracts/schemas` exactly.
/// If a contract enum changes, update the matching enum here and the contract test will confirm it.
library;

abstract interface class WireEnum {
  String get wire;
}

T enumFromWire<T extends WireEnum>(List<T> values, Object? value, {T? fallback}) {
  for (final v in values) {
    if (v.wire == value) return v;
  }
  if (fallback != null) return fallback;
  throw FormatException('Unknown value "$value" for $T');
}

enum TaskCategory implements WireEnum {
  study('study', 'Study'),
  work('work', 'Work'),
  exercise('exercise', 'Exercise'),
  chores('chores', 'Chores'),
  personal('personal', 'Personal'),
  social('social', 'Social'),
  other('other', 'Other');

  const TaskCategory(this.wire, this.label);
  @override
  final String wire;
  final String label;

  static TaskCategory fromWire(Object? v) => enumFromWire(values, v, fallback: TaskCategory.other);
}

enum TaskStatus implements WireEnum {
  pending('pending', 'Pending'),
  scheduled('scheduled', 'Scheduled'),
  done('done', 'Done'),
  missed('missed', 'Missed'),
  cancelled('cancelled', 'Cancelled');

  const TaskStatus(this.wire, this.label);
  @override
  final String wire;
  final String label;

  static TaskStatus fromWire(Object? v) => enumFromWire(values, v, fallback: TaskStatus.pending);
}

/// OPEN ISSUE (team): Module 1 may move from five to three classes — update here and in contracts.
enum AffectiveLabel implements WireEnum {
  stress('stress', 'Stressed'),
  relaxed('relaxed', 'Relaxed'),
  happy('happy', 'Happy'),
  sad('sad', 'Sad'),
  anxiety('anxiety', 'Anxious');

  const AffectiveLabel(this.wire, this.label);
  @override
  final String wire;
  final String label;

  static AffectiveLabel fromWire(Object? v) => enumFromWire(values, v);
}

enum MealType implements WireEnum {
  breakfast('breakfast', 'Breakfast'),
  lunch('lunch', 'Lunch'),
  dinner('dinner', 'Dinner'),
  snack('snack', 'Snack');

  const MealType(this.wire, this.label);
  @override
  final String wire;
  final String label;

  static MealType fromWire(Object? v) => enumFromWire(values, v, fallback: MealType.snack);
}

enum KeyClass implements WireEnum {
  letter('letter'),
  digit('digit'),
  space('space'),
  backspace('backspace'),
  enter('enter'),
  punctuation('punctuation'),
  other('other');

  const KeyClass(this.wire);
  @override
  final String wire;

  static KeyClass fromWire(Object? v) => enumFromWire(values, v, fallback: KeyClass.other);
}

enum ScreenEventType implements WireEnum {
  screenOn('screen_on'),
  screenOff('screen_off'),
  unlock('unlock');

  const ScreenEventType(this.wire);
  @override
  final String wire;

  static ScreenEventType fromWire(Object? v) => enumFromWire(values, v);
}

enum AppCategory implements WireEnum {
  game('game'),
  audio('audio'),
  video('video'),
  image('image'),
  social('social'),
  news('news'),
  maps('maps'),
  productivity('productivity'),
  accessibility('accessibility'),
  unknown('unknown');

  const AppCategory(this.wire);
  @override
  final String wire;

  static AppCategory fromWire(Object? v) => enumFromWire(values, v, fallback: AppCategory.unknown);
}

enum ActivityType implements WireEnum {
  still('still'),
  walking('walking'),
  running('running'),
  onBicycle('on_bicycle'),
  inVehicle('in_vehicle'),
  unknown('unknown');

  const ActivityType(this.wire);
  @override
  final String wire;

  static ActivityType fromWire(Object? v) => enumFromWire(values, v, fallback: ActivityType.unknown);
}
