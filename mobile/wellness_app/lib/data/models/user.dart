import '../../core/utils/json.dart';

/// `User` from the core API (`GET /v1/me`).
class User {
  const User({
    required this.id,
    required this.email,
    required this.displayName,
    required this.timezone,
    required this.createdAt,
    this.consentVersion,
    this.consentedAt,
  });

  factory User.fromJson(JsonMap json) => User(
        id: json['id'] as String,
        email: json['email'] as String,
        displayName: json['display_name'] as String,
        timezone: json['timezone'] as String,
        createdAt: parseDate(json['created_at']),
        consentVersion: json['consent_version'] as String?,
        consentedAt: parseDateOrNull(json['consented_at']),
      );

  final String id;
  final String email;
  final String displayName;
  final String timezone;
  final DateTime createdAt;
  final String? consentVersion;
  final DateTime? consentedAt;

  JsonMap toJson() => {
        'id': id,
        'email': email,
        'display_name': displayName,
        'timezone': timezone,
        'created_at': dateToJson(createdAt),
        'consent_version': consentVersion,
        'consented_at': dateToJsonOrNull(consentedAt),
      };
}
