import 'package:shared_preferences/shared_preferences.dart';

/// App-wide constants and user-editable settings persisted in SharedPreferences.
class AppConfig {
  const AppConfig({required this.serverUrl});

  /// Default works on an emulator or USB-connected phone after `adb reverse tcp:8000 tcp:8000`.
  static const defaultServerUrl = 'http://127.0.0.1:8000';
  static const appVersion = '0.1.0';

  /// Bump when the consent text in `ConsentScreen` changes; users are asked to consent again.
  static const consentVersion = '2026-10-v1';

  static const defaultTimezone = 'Asia/Colombo';

  // SharedPreferences keys (also read by the background isolate — keep them stable).
  static const kServerUrl = 'server_url';
  static const kDeviceId = 'device_id';
  static const kTimezone = 'timezone';
  static const kSensingDir = 'sensing_dir';
  static const kLastCollectedAt = 'last_collected_at_ms';
  static const kLastUploadAt = 'last_upload_at_ms';
  static const kLastSteps = 'last_step_counter';
  static const kLastStepsAt = 'last_step_counter_at_ms';

  final String serverUrl;

  static AppConfig load(SharedPreferences prefs) =>
      AppConfig(serverUrl: prefs.getString(kServerUrl) ?? defaultServerUrl);

  /// Strips trailing slashes so `baseUrl + '/v1/...'` never doubles them.
  static String normaliseUrl(String url) => url.trim().replaceAll(RegExp(r'/+$'), '');
}
