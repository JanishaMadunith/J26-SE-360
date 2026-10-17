import '../../core/utils/json.dart';
import 'user.dart';

class RegisterRequest {
  const RegisterRequest({
    required this.email,
    required this.password,
    required this.displayName,
    required this.timezone,
  });

  final String email;
  final String password;
  final String displayName;
  final String timezone;

  JsonMap toJson() => {
        'email': email,
        'password': password,
        'display_name': displayName,
        'timezone': timezone,
      };
}

class LoginRequest {
  const LoginRequest({required this.email, required this.password});

  final String email;
  final String password;

  JsonMap toJson() => {'email': email, 'password': password};
}

/// Response of register / login / refresh.
class TokenResponse {
  const TokenResponse({
    required this.accessToken,
    required this.refreshToken,
    required this.tokenType,
    this.user,
  });

  factory TokenResponse.fromJson(JsonMap json) => TokenResponse(
        accessToken: json['access_token'] as String,
        refreshToken: json['refresh_token'] as String,
        tokenType: (json['token_type'] as String?) ?? 'bearer',
        user: json['user'] == null ? null : User.fromJson(json['user'] as JsonMap),
      );

  final String accessToken;
  final String refreshToken;
  final String tokenType;
  final User? user;

  JsonMap toJson() => {
        'access_token': accessToken,
        'refresh_token': refreshToken,
        'token_type': tokenType,
        if (user != null) 'user': user!.toJson(),
      };
}
