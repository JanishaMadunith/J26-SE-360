import 'package:dio/dio.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../data/api/apis.dart';
import '../features/auth/auth_controller.dart';
import 'config/app_config.dart';
import 'network/api_client.dart';
import 'storage/token_store.dart';

/// Overridden in `main()` with the instance loaded before `runApp`.
final sharedPreferencesProvider = Provider<SharedPreferences>(
  (ref) => throw UnimplementedError('sharedPreferencesProvider must be overridden in main()'),
);

class AppConfigNotifier extends Notifier<AppConfig> {
  @override
  AppConfig build() => AppConfig.load(ref.watch(sharedPreferencesProvider));

  Future<void> setServerUrl(String url) async {
    final clean = AppConfig.normaliseUrl(url);
    await ref.read(sharedPreferencesProvider).setString(AppConfig.kServerUrl, clean);
    state = AppConfig(serverUrl: clean);
  }
}

final appConfigProvider = NotifierProvider<AppConfigNotifier, AppConfig>(AppConfigNotifier.new);

final tokenStoreProvider = Provider<TokenStore>((ref) => TokenStore());

final dioProvider = Provider<Dio>((ref) {
  final config = ref.watch(appConfigProvider);
  return createDio(
    baseUrl: config.serverUrl,
    tokens: ref.watch(tokenStoreProvider),
    onSessionExpired: () => ref.read(authControllerProvider.notifier).onSessionExpired(),
  );
});

final authApiProvider = Provider((ref) => AuthApi(ref.watch(dioProvider)));
final meApiProvider = Provider((ref) => MeApi(ref.watch(dioProvider)));
final sensingApiProvider = Provider((ref) => SensingApi(ref.watch(dioProvider)));
final tasksApiProvider = Provider((ref) => TasksApi(ref.watch(dioProvider)));
final scheduleApiProvider = Provider((ref) => ScheduleApi(ref.watch(dioProvider)));
final insightsApiProvider = Provider((ref) => InsightsApi(ref.watch(dioProvider)));
final recommendationsApiProvider = Provider((ref) => RecommendationsApi(ref.watch(dioProvider)));
final pipelineApiProvider = Provider((ref) => PipelineApi(ref.watch(dioProvider)));
final healthApiProvider = Provider((ref) => HealthApi(ref.watch(dioProvider)));
