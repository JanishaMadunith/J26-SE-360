import '../../core/utils/json.dart';

/// Which model produced a result (`ModelInfo` contract).
class ModelInfo {
  const ModelInfo({required this.name, required this.version});

  factory ModelInfo.fromJson(JsonMap json) =>
      ModelInfo(name: json['name'] as String, version: json['version'] as String);

  final String name;
  final String version;

  JsonMap toJson() => {'name': name, 'version': version};

  @override
  String toString() => '$name $version';
}
