import 'package:light/light.dart';

import '../../data/models/sensing.dart';

/// Takes a few ambient-light readings (lux). Used server-side for sleep estimation.
class LightCollector {
  Future<List<LightReading>> sample({int count = 3, Duration timeout = const Duration(seconds: 4)}) async {
    final readings = <LightReading>[];
    try {
      // UNVERIFIED API: `light` exposes `Light().lightSensorStream` (Stream<int>) in v3/v4.
      await for (final lux in Light().lightSensorStream.timeout(timeout)) {
        readings.add(LightReading(at: DateTime.now(), lux: lux.toDouble()));
        if (readings.length >= count) break;
      }
    } catch (_) {
      // No light sensor or timed out — return whatever we have.
    }
    return readings;
  }
}
