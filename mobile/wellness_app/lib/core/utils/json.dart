/// JSON helpers shared by every model.
///
/// The backend (Pydantic `AwareDatetime`) requires timezone-aware timestamps, so we always send
/// UTC ISO-8601 strings ending in `Z` and convert to local time when reading.
library;

typedef JsonMap = Map<String, dynamic>;

DateTime parseDate(Object? value) => DateTime.parse(value! as String).toLocal();

DateTime? parseDateOrNull(Object? value) => value == null ? null : parseDate(value);

String dateToJson(DateTime value) => value.toUtc().toIso8601String();

String? dateToJsonOrNull(DateTime? value) => value == null ? null : dateToJson(value);

DateTime fromEpochMs(Object? value) => DateTime.fromMillisecondsSinceEpoch((value! as num).toInt());

double toDouble(Object? value) => (value! as num).toDouble();

double? toDoubleOrNull(Object? value) => value == null ? null : (value as num).toDouble();

int toInt(Object? value) => (value! as num).toInt();

int? toIntOrNull(Object? value) => value == null ? null : (value as num).toInt();

List<T> parseList<T>(Object? value, T Function(JsonMap json) fromJson) =>
    ((value as List<dynamic>?) ?? const <dynamic>[]).map((e) => fromJson(e as JsonMap)).toList();

List<double> parseDoubles(Object? value) =>
    ((value as List<dynamic>?) ?? const <dynamic>[]).map((e) => (e as num).toDouble()).toList();

List<int> parseInts(Object? value) =>
    ((value as List<dynamic>?) ?? const <dynamic>[]).map((e) => (e as num).toInt()).toList();

List<String> parseStrings(Object? value) =>
    ((value as List<dynamic>?) ?? const <dynamic>[]).map((e) => e as String).toList();
