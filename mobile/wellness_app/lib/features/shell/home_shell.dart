import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../core/router/app_router.dart';
import '../../sensing/sensing_providers.dart';

/// Bottom-navigation shell. Also triggers a foreground sync when the app opens or resumes
/// (at most every 5 minutes) and keeps the activity-recognition listener running.
class HomeShell extends ConsumerStatefulWidget {
  const HomeShell({required this.child, super.key});

  final Widget child;

  @override
  ConsumerState<HomeShell> createState() => _HomeShellState();
}

class _HomeShellState extends ConsumerState<HomeShell> with WidgetsBindingObserver {
  static const _tabs = [
    (Routes.home, Icons.dashboard_outlined, Icons.dashboard, 'Today'),
    (Routes.tasks, Icons.checklist_outlined, Icons.checklist, 'Tasks'),
    (Routes.schedule, Icons.calendar_today_outlined, Icons.calendar_today, 'Schedule'),
    (Routes.meals, Icons.restaurant_outlined, Icons.restaurant, 'Meals'),
    (Routes.more, Icons.more_horiz, Icons.more_horiz, 'More'),
  ];

  DateTime? _lastSync;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    WidgetsBinding.instance.addPostFrameCallback((_) {
      unawaited(ref.read(activityCollectorProvider).start());
      unawaited(_backgroundSync());
    });
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed) unawaited(_backgroundSync());
  }

  Future<void> _backgroundSync() async {
    final now = DateTime.now();
    if (_lastSync != null && now.difference(_lastSync!) < const Duration(minutes: 5)) return;
    _lastSync = now;
    try {
      await syncNow(ref);
    } catch (_) {
      // Offline or server down — the queue retries later.
    }
  }

  int _indexFor(String location) {
    for (var i = _tabs.length - 1; i > 0; i--) {
      if (location.startsWith(_tabs[i].$1)) return i;
    }
    return 0;
  }

  @override
  Widget build(BuildContext context) {
    final location = GoRouterState.of(context).matchedLocation;
    final index = _indexFor(location);
    return Scaffold(
      body: widget.child,
      bottomNavigationBar: NavigationBar(
        selectedIndex: index,
        onDestinationSelected: (i) => context.go(_tabs[i].$1),
        destinations: [
          for (final t in _tabs) NavigationDestination(icon: Icon(t.$2), selectedIcon: Icon(t.$3), label: t.$4),
        ],
      ),
    );
  }
}
