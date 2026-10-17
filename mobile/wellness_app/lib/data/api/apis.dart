/// One small API class per core-API area (see BUILD_SPEC / docs/api-contracts.md).
/// Every method throws [ApiException] on failure.
library;

import 'package:dio/dio.dart';

import '../../core/network/api_client.dart';
import '../../core/utils/json.dart';
import '../models/auth.dart';
import '../models/diet_profile.dart';
import '../models/insights.dart';
import '../models/pipeline.dart';
import '../models/recommendations.dart';
import '../models/schedule.dart';
import '../models/sensing.dart';
import '../models/task.dart';
import '../models/user.dart';
import '../models/wire_enums.dart';

Future<T> _call<T>(Future<T> Function() fn) async {
  try {
    return await fn();
  } catch (e) {
    throw ApiException.from(e);
  }
}

JsonMap _map(Response<dynamic> r) => r.data as JsonMap;

class AuthApi {
  AuthApi(this._dio);
  final Dio _dio;

  Future<TokenResponse> register(RegisterRequest body) => _call(() async {
        final r = await _dio.post<dynamic>('/v1/auth/register',
            data: body.toJson(), options: Options(extra: {kSkipAuth: true}));
        return TokenResponse.fromJson(_map(r));
      });

  Future<TokenResponse> login(LoginRequest body) => _call(() async {
        final r = await _dio.post<dynamic>('/v1/auth/login',
            data: body.toJson(), options: Options(extra: {kSkipAuth: true}));
        return TokenResponse.fromJson(_map(r));
      });
}

class MeApi {
  MeApi(this._dio);
  final Dio _dio;

  Future<User> me() => _call(() async => User.fromJson(_map(await _dio.get<dynamic>('/v1/me'))));

  Future<User> consent(String version) => _call(
        () async => User.fromJson(_map(await _dio.post<dynamic>('/v1/me/consent', data: {'version': version}))),
      );

  Future<DietProfile> dietProfile() =>
      _call(() async => DietProfile.fromJson(_map(await _dio.get<dynamic>('/v1/me/diet-profile'))));

  Future<DietProfile> saveDietProfile(DietProfile profile) => _call(
        () async => DietProfile.fromJson(_map(await _dio.put<dynamic>('/v1/me/diet-profile', data: profile.toJson()))),
      );
}

/// Result of trying to upload one queued batch (drives the offline queue's retry logic).
enum SendOutcome { sent, duplicate, rejected, retry }

class SensingApi {
  SensingApi(this._dio);
  final Dio _dio;

  /// Posts an already-serialised [SensorBatch] JSON string. Never throws.
  Future<SendOutcome> uploadBatchJson(String payload) async {
    try {
      final r = await _dio.post<dynamic>('/v1/sensing/batches', data: payload);
      final ack = SensorBatchAck.fromJson(_map(r));
      return ack.duplicate ? SendOutcome.duplicate : SendOutcome.sent;
    } on DioException catch (e) {
      final status = e.response?.statusCode;
      if (status == 409) return SendOutcome.duplicate;
      if (status == 400 || status == 413 || status == 422) return SendOutcome.rejected;
      return SendOutcome.retry; // network error, 401 after failed refresh, 5xx
    } catch (_) {
      return SendOutcome.retry;
    }
  }

  Future<void> postSelfReport(SelfReport report) => _call<void>(() async {
        await _dio.post<dynamic>('/v1/self-reports', data: report.toJson());
      });
}

class TasksApi {
  TasksApi(this._dio);
  final Dio _dio;

  Future<List<Task>> list({TaskStatus? status}) => _call(() async {
        final r = await _dio.get<dynamic>(
          '/v1/tasks',
          queryParameters: {if (status != null) 'status': status.wire},
        );
        return parseList(r.data, Task.fromJson);
      });

  Future<Task> create(TaskCreate body) =>
      _call(() async => Task.fromJson(_map(await _dio.post<dynamic>('/v1/tasks', data: body.toJson()))));

  Future<Task> update(String id, TaskUpdate body) =>
      _call(() async => Task.fromJson(_map(await _dio.patch<dynamic>('/v1/tasks/$id', data: body.toJson()))));

  Future<Task> complete(String id) =>
      _call(() async => Task.fromJson(_map(await _dio.post<dynamic>('/v1/tasks/$id/complete'))));

  Future<void> delete(String id) => _call<void>(() async {
        await _dio.delete<dynamic>('/v1/tasks/$id');
      });
}

class ScheduleApi {
  ScheduleApi(this._dio);
  final Dio _dio;

  /// Returns null when no schedule exists yet (404).
  Future<ScheduleResult?> current() async {
    try {
      return ScheduleResult.fromJson(_map(await _dio.get<dynamic>('/v1/schedule/current')));
    } catch (e) {
      final ex = ApiException.from(e);
      if (ex.isNotFound) return null;
      throw ex;
    }
  }

  Future<ScheduleResult> regenerate() =>
      _call(() async => ScheduleResult.fromJson(_map(await _dio.post<dynamic>('/v1/schedule/regenerate'))));
}

class InsightsApi {
  InsightsApi(this._dio);
  final Dio _dio;

  Future<Insights> latest() => _call(() async => Insights.fromJson(_map(await _dio.get<dynamic>('/v1/insights/latest'))));
}

class RecommendationsApi {
  RecommendationsApi(this._dio);
  final Dio _dio;

  /// Returns null when there are no recommendations yet (404).
  Future<Recommendations?> latest() async {
    try {
      return Recommendations.fromJson(_map(await _dio.get<dynamic>('/v1/recommendations/latest')));
    } catch (e) {
      final ex = ApiException.from(e);
      if (ex.isNotFound) return null;
      throw ex;
    }
  }
}

class PipelineApi {
  PipelineApi(this._dio);
  final Dio _dio;

  Future<PipelineRun> run() =>
      _call(() async => PipelineRun.fromJson(_map(await _dio.post<dynamic>('/v1/pipeline/run'))));

  Future<List<PipelineRun>> runs({int limit = 10}) => _call(() async {
        final r = await _dio.get<dynamic>('/v1/pipeline/runs', queryParameters: {'limit': limit});
        return parseList(r.data, PipelineRun.fromJson);
      });
}

class HealthApi {
  HealthApi(this._dio);
  final Dio _dio;

  /// Raw `/health` body (`status`, `db`, `modules`).
  Future<JsonMap> health() => _call(() async => _map(await _dio.get<dynamic>('/health', options: Options(extra: {kSkipAuth: true}))));
}
