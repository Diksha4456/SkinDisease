import 'package:flutter_test/flutter_test.dart';
import 'package:fairskin_mobile/main.dart';

void main() {
  testWidgets('FairSkinApp smoke test', (WidgetTester tester) async {
    await tester.pumpWidget(const FairSkinApp());
    expect(find.text('FairSkin AI Screening'), findsOneWidget);
  });
}
