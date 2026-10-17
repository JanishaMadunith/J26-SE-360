import 'package:dio/dio.dart';

import '../storage/token_store.dart';

/// Error shown to the user. Built from FastAPI's `{"detail": ...}` error bodies when available.
class ApiException implements Exception {
  ApiException(this.message, {this.statusCode});

  factory ApiException.from(Object error) {
    if (error is ApiException) return error;
    if (error is DioException) {
      final status = error.response?.statusCode;
      final data = error.response?.data;
      String? detail;
      if (data is Map<String, dynamic>) {
        final d = data['detail'];
        if (d is String) {
          detail = d;
        } else if (d is List<dynamic> && d.isNotEmpty) {
          final first = d.first;
          detail = first is Map<String, dynamic> ? '${first['msg']}' : '$first';
        }
      }
      if (detail != null) return ApiException(detail, statusCode: status);
      switch (error.type) {
        case DioExceptionType.connectionTimeout:
        case DioExceptionType.receiveTimeout:
        case DioExceptionType.sendTimeout:
          return ApiException('The server took too long to respond.', statusCode: status);
        case DioExceptionType.connectionError:
          return ApiException('Cannot reach the server. Check the server URL in Settings.', statusCode: status);
        default:
          return ApiException(status == null ? 'Network error' : 'Server error ($status)', statusCode: status);
      }
    }
    return ApiException(error.toString());
  }

  final String message;
  final int? statusCode;

  bool get isNotFound => statusCode == 404;
  bool get isUnauthorized => statusCode == 401;

  @override
  String toString() => message;
}

/// Request option flag: do not attach the bearer token (login / register / refresh).
const kSkipAuth = 'skipAuth';

Dio createDio({
  required String baseUrl,
  required TokenStore tokens,
  void Function()? onSessionExpired,
}) {
  final dio = Dio(
    BaseOptions(
      baseUrl: baseUrl,
      connectTimeout: const Duration(seconds: 10),
      receiveTimeout: const Duration(seconds: 30),
      sendTimeout: const Duration(seconds: 60),
      contentType: Headers.jsonContentType,
      responseType: ResponseType.json,
    ),
  );
  dio.interceptors.add(AuthInterceptor(dio: dio, tokens: tokens, onSessionExpired: onSessionExpired));
  return dio;
}

/// Adds `Authorization: Bearer <access>` and transparently refreshes once on a 401.
class AuthInterceptor extends QueuedInterceptor {
  AuthInterceptor({required this.dio, required this.tokens, this.onSessionExpired});

  final Dio dio;
  final TokenStore tokens;
  final void Function()? onSessionExpired;

  @override
  Future<void> onRequest(RequestOptions options, RequestInterceptorHandler handler) async {
    if (options.extra[kSkipAuth] != true) {
      final access = await tokens.accessToken();
      if (access != null) options.headers['Authorization'] = 'Bearer $access';
    }
    handler.next(options);
  }

  @override
  Future<void> onError(DioException err, ErrorInterceptorHandler handler) async {
    final request = err.requestOptions;
    final unauthorized = err.response?.statusCode == 401;
    if (!unauthorized || request.extra[kSkipAuth] == true || request.extra['retried'] == true) {
      handler.next(err);
      return;
    }
    final refresh = await tokens.refreshToken();
    if (refresh == null) {
      onSessionExpired?.call();
      handler.next(err);
      return;
    }
    try {
      final plain = Dio(BaseOptions(baseUrl: dio.options.baseUrl, contentType: Headers.jsonContentType));
      final res = await plain.post<Map<String, dynamic>>(
        '/v1/auth/refresh',
        data: <String, dynamic>{'refresh_token': refresh},
      );
      final body = res.data!;
      final access = body['access_token'] as String;
      await tokens.save(access: access, refresh: body['refresh_token'] as String);
      request.headers['Authorization'] = 'Bearer $access';
      request.extra['retried'] = true;
      final retried = await dio.fetch<dynamic>(request);
      handler.resolve(retried);
    } catch (_) {
      await tokens.clear();
      onSessionExpired?.call();
      handler.next(err);
    }
  }
}
