import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../data/models/task.dart';
import '../../features/auth/auth_controller.dart';
import '../../features/auth/login_screen.dart';
import '../../features/auth/register_screen.dart';
import '../../features/check_in/check_in_screen.dart';
import '../../features/home_dashboard/dashboard_screen.dart';
import '../../features/meals/meals_screen.dart';
import '../../features/onboarding_consent/consent_screen.dart';
import '../../features/ppg_measure/ppg_screen.dart';
import '../../features/schedule/schedule_screen.dart';
import '../../features/settings/more_screen.dart';
import '../../features/settings/settings_screen.dart';
import '../../features/shell/home_shell.dart';
import '../../features/tasks/task_form_screen.dart';
import '../../features/tasks/tasks_screen.dart';

abstract final class Routes {
  static const splash = '/splash';
  static const login = '/login';
  static const register = '/register';
  static const consent = '/consent';
  static const home = '/';
  static const tasks = '/tasks';
  static const newTask = '/tasks/new';
  static const editTask = '/tasks/edit';
  static const schedule = '/schedule';
  static const meals = '/meals';
  static const more = '/more';
  static const checkIn = '/check-in';
  static const ppg = '/ppg';
  static const settings = '/settings';
}

final routerProvider = Provider<GoRouter>((ref) {
  // Re-run redirects whenever the auth state changes.
  final refresh = ValueNotifier<int>(0);
  ref.listen(authControllerProvider, (_, __) => refresh.value++);
  ref.onDispose(refresh.dispose);

  return GoRouter(
    initialLocation: Routes.home,
    refreshListenable: refresh,
    redirect: (context, state) {
      final auth = ref.read(authControllerProvider);
      final loc = state.matchedLocation;
      if (!auth.hasValue) return loc == Routes.splash ? null : Routes.splash;
      final s = auth.valueOrNull ?? const AuthState();
      final atAuth = loc == Routes.login || loc == Routes.register;
      // Settings stays reachable while logged out so the server URL can be fixed before logging in.
      if (!s.isLoggedIn) return (atAuth || loc == Routes.settings) ? null : Routes.login;
      if (!s.hasConsented) return loc == Routes.consent ? null : Routes.consent;
      if (atAuth || loc == Routes.consent || loc == Routes.splash) return Routes.home;
      return null;
    },
    routes: [
      GoRoute(path: Routes.splash, builder: (_, __) => const _SplashScreen()),
      GoRoute(path: Routes.login, builder: (_, __) => const LoginScreen()),
      GoRoute(path: Routes.register, builder: (_, __) => const RegisterScreen()),
      GoRoute(path: Routes.consent, builder: (_, __) => const ConsentScreen()),
      ShellRoute(
        builder: (context, state, child) => HomeShell(child: child),
        routes: [
          GoRoute(path: Routes.home, builder: (_, __) => const DashboardScreen()),
          GoRoute(path: Routes.tasks, builder: (_, __) => const TasksScreen()),
          GoRoute(path: Routes.schedule, builder: (_, __) => const ScheduleScreen()),
          GoRoute(path: Routes.meals, builder: (_, __) => const MealsScreen()),
          GoRoute(path: Routes.more, builder: (_, __) => const MoreScreen()),
        ],
      ),
      GoRoute(path: Routes.newTask, builder: (_, __) => const TaskFormScreen()),
      GoRoute(
        path: Routes.editTask,
        builder: (_, state) => TaskFormScreen(existing: state.extra as Task?),
      ),
      GoRoute(path: Routes.checkIn, builder: (_, __) => const CheckInScreen()),
      GoRoute(path: Routes.ppg, builder: (_, __) => const PpgScreen()),
      GoRoute(path: Routes.settings, builder: (_, __) => const SettingsScreen()),
    ],
  );
});

class _SplashScreen extends StatelessWidget {
  const _SplashScreen();

  @override
  Widget build(BuildContext context) => const Scaffold(body: Center(child: CircularProgressIndicator()));
}
