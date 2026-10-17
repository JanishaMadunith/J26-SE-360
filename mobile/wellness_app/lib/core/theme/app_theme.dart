import 'package:flutter/material.dart';

/// Material 3 theme generated from one seed colour (calm teal) for light and dark mode.
class AppTheme {
  static const _seed = Color(0xFF14877A);

  static ThemeData light() => _base(ColorScheme.fromSeed(seedColor: _seed));

  static ThemeData dark() => _base(ColorScheme.fromSeed(seedColor: _seed, brightness: Brightness.dark));

  static ThemeData _base(ColorScheme scheme) => ThemeData(
        colorScheme: scheme,
        useMaterial3: true,
        visualDensity: VisualDensity.standard,
        inputDecorationTheme: const InputDecorationTheme(border: OutlineInputBorder()),
      );

  /// Colour for a completion probability or score (high = good).
  static Color scoreColor(ColorScheme scheme, double value) {
    if (value >= 0.7) return scheme.primary;
    if (value >= 0.4) return scheme.tertiary;
    return scheme.error;
  }
}
