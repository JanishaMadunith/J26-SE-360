// Smoke test. Also prevents `flutter create .` from generating its default counter-app test,
// which would not compile against this project.
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:wellness_app/features/onboarding_consent/consent_screen.dart';

void main() {
  testWidgets('consent screen renders and Continue is disabled until the box is ticked', (tester) async {
    await tester.pumpWidget(const ProviderScope(child: MaterialApp(home: ConsentScreen())));
    expect(find.text('What is collected'), findsOneWidget);
    final button = tester.widget<FilledButton>(find.widgetWithText(FilledButton, 'Continue'));
    expect(button.onPressed, isNull);
    await tester.tap(find.byType(CheckboxListTile));
    await tester.pump();
    expect(tester.widget<FilledButton>(find.widgetWithText(FilledButton, 'Continue')).onPressed, isNotNull);
  });
}
