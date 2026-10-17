import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/config/app_config.dart';
import '../../core/providers.dart';
import '../../data/models/auth.dart';
import '../../data/models/user.dart';

class AuthState {
  const AuthState({this.user});

  final User? user;

  bool get isLoggedIn => user != null;

  /// True once the user accepted the CURRENT consent version.
  bool get hasConsented => user?.consentVersion == AppConfig.consentVersion;
}

/// Holds who is logged in. Login/register errors are rethrown to the calling screen
/// (the global state is not put into an error state, so routing stays stable).
class AuthController extends AsyncNotifier<AuthState> {
  @override
  Future<AuthState> build() async {
    final tokens = ref.read(tokenStoreProvider);
    if (await tokens.accessToken() == null) return const AuthState();
    try {
      final me = await ref.read(meApiProvider).me();
      return AuthState(user: me);
    } catch (_) {
      // Token expired and refresh failed, or server unreachable: ask to log in again.
      return const AuthState();
    }
  }

  Future<void> login({required String email, required String password}) async {
    final res = await ref.read(authApiProvider).login(LoginRequest(email: email, password: password));
    await _signedIn(res);
  }

  Future<void> register({
    required String email,
    required String password,
    required String displayName,
    required String timezone,
  }) async {
    final res = await ref.read(authApiProvider).register(
          RegisterRequest(email: email, password: password, displayName: displayName, timezone: timezone),
        );
    await _signedIn(res);
  }

  Future<void> acceptConsent() async {
    final user = await ref.read(meApiProvider).consent(AppConfig.consentVersion);
    state = AsyncData(AuthState(user: user));
  }

  Future<void> refreshUser() async {
    final user = await ref.read(meApiProvider).me();
    state = AsyncData(AuthState(user: user));
  }

  Future<void> logout() async {
    await ref.read(tokenStoreProvider).clear();
    state = const AsyncData(AuthState());
  }

  void onSessionExpired() {
    state = const AsyncData(AuthState());
  }

  Future<void> _signedIn(TokenResponse res) async {
    await ref.read(tokenStoreProvider).save(access: res.accessToken, refresh: res.refreshToken);
    final user = res.user ?? await ref.read(meApiProvider).me();
    state = AsyncData(AuthState(user: user));
  }
}

final authControllerProvider = AsyncNotifierProvider<AuthController, AuthState>(AuthController.new);
