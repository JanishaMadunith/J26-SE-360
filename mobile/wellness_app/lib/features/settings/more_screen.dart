import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../core/router/app_router.dart';
import '../auth/auth_controller.dart';

class MoreScreen extends ConsumerWidget {
  const MoreScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final user = ref.watch(authControllerProvider).valueOrNull?.user;
    return Scaffold(
      appBar: AppBar(title: const Text('More')),
      body: ListView(
        children: [
          if (user != null)
            ListTile(
              leading: const CircleAvatar(child: Icon(Icons.person)),
              title: Text(user.displayName),
              subtitle: Text('${user.email} · ${user.timezone}'),
            ),
          const Divider(),
          ListTile(
            leading: const Icon(Icons.mood),
            title: const Text('Check in'),
            subtitle: const Text('Rate your mood, stress, energy and focus'),
            onTap: () => context.push(Routes.checkIn),
          ),
          ListTile(
            leading: const Icon(Icons.favorite_outline),
            title: const Text('Measure heart rate'),
            subtitle: const Text('60-second fingertip measurement with the camera'),
            onTap: () => context.push(Routes.ppg),
          ),
          ListTile(
            leading: const Icon(Icons.settings_outlined),
            title: const Text('Settings & permissions'),
            onTap: () => context.push(Routes.settings),
          ),
        ],
      ),
    );
  }
}
