import 'package:permission_handler/permission_handler.dart';

import '../../sensing/native_bridge.dart';

/// Status of every permission the sensing features need.
class SensingPermissions {
  const SensingPermissions({
    required this.usageAccess,
    required this.typingService,
    required this.activity,
    required this.notifications,
    required this.camera,
  });

  final bool usageAccess;
  final bool typingService;
  final bool activity;
  final bool notifications;
  final bool camera;

  int get grantedCount => [usageAccess, typingService, activity, notifications, camera].where((g) => g).length;
}

class PermissionsService {
  PermissionsService(this._native);

  final NativeBridge _native;

  Future<SensingPermissions> status() async {
    Future<bool> safe(Future<bool> Function() f) async {
      try {
        return await f();
      } catch (_) {
        return false;
      }
    }

    return SensingPermissions(
      usageAccess: await safe(_native.hasUsageAccess),
      typingService: await safe(_native.isTypingServiceEnabled),
      activity: await safe(() async => (await Permission.activityRecognition.status).isGranted),
      notifications: await safe(() async => (await Permission.notification.status).isGranted),
      camera: await safe(() async => (await Permission.camera.status).isGranted),
    );
  }

  /// Usage access is a special app-op: the user must flip it in system settings.
  Future<void> requestUsageAccess() => _native.openUsageAccessSettings();

  /// The typing-rhythm accessibility service must be enabled by the user in system settings.
  Future<void> requestTypingService() => _native.openAccessibilitySettings();

  Future<void> requestActivity() async {
    await Permission.activityRecognition.request();
  }

  Future<void> requestNotifications() async {
    await Permission.notification.request();
  }

  Future<void> requestCamera() async {
    await Permission.camera.request();
  }
}
