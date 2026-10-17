import 'dart:typed_data';

/// Mean colour of one camera frame.
class RgbMean {
  const RgbMean(this.r, this.g, this.b);

  final double r;
  final double g;
  final double b;

  @override
  String toString() => 'RgbMean(${r.toStringAsFixed(1)}, ${g.toStringAsFixed(1)}, ${b.toStringAsFixed(1)})';
}

/// Mean R/G/B of the centre of a YUV_420_888 frame (Android camera image stream).
///
/// Because the YUV→RGB conversion is linear, the RGB of the mean equals the mean of the RGB (before
/// clipping), so we average Y, U and V and convert once — cheap enough to run on every frame.
/// Only these three numbers per frame leave the phone; the frame itself is discarded.
///
/// * [yPlane] has pixel stride 1 (always true for YUV_420_888).
/// * [uPlane]/[vPlane] are 2×2 subsampled; [uvPixelStride] is 1 (planar) or 2 (semi-planar/interleaved).
/// * [step] samples every n-th pixel in both directions; [cropFraction] keeps the centre region.
RgbMean yuv420Mean({
  required int width,
  required int height,
  required Uint8List yPlane,
  required int yRowStride,
  required Uint8List uPlane,
  required Uint8List vPlane,
  required int uvRowStride,
  required int uvPixelStride,
  int step = 4,
  double cropFraction = 0.5,
}) {
  final x0 = (width * (1 - cropFraction) / 2).floor();
  final y0 = (height * (1 - cropFraction) / 2).floor();
  final x1 = width - x0;
  final y1 = height - y0;
  var sumY = 0.0;
  var sumU = 0.0;
  var sumV = 0.0;
  var n = 0;
  for (var y = y0; y < y1; y += step) {
    final yRow = y * yRowStride;
    final uvRow = (y >> 1) * uvRowStride;
    for (var x = x0; x < x1; x += step) {
      final yi = yRow + x;
      final uvi = uvRow + (x >> 1) * uvPixelStride;
      if (yi >= yPlane.length || uvi >= uPlane.length || uvi >= vPlane.length) continue;
      sumY += yPlane[yi];
      sumU += uPlane[uvi];
      sumV += vPlane[uvi];
      n++;
    }
  }
  if (n == 0) return const RgbMean(0, 0, 0);
  final my = sumY / n;
  final mu = sumU / n - 128;
  final mv = sumV / n - 128;
  double clamp(double v) => v < 0 ? 0 : (v > 255 ? 255 : v);
  return RgbMean(
    clamp(my + 1.402 * mv),
    clamp(my - 0.344136 * mu - 0.714136 * mv),
    clamp(my + 1.772 * mu),
  );
}

/// Rough live heart-rate estimate for on-screen feedback only (the server computes the real value):
/// counts peaks of the de-trended red signal.
double? roughBpm(List<double> red, List<int> tMs) {
  if (red.length < 90 || tMs.length != red.length) return null;
  const window = 15;
  final detrended = <double>[];
  for (var i = 0; i < red.length; i++) {
    final lo = i - window < 0 ? 0 : i - window;
    final hi = i + window >= red.length ? red.length - 1 : i + window;
    var s = 0.0;
    for (var j = lo; j <= hi; j++) {
      s += red[j];
    }
    detrended.add(red[i] - s / (hi - lo + 1));
  }
  var peaks = 0;
  var lastPeak = -1000000;
  for (var i = 1; i < detrended.length - 1; i++) {
    final isPeak = detrended[i] > 0 && detrended[i] > detrended[i - 1] && detrended[i] >= detrended[i + 1];
    if (isPeak && tMs[i] - lastPeak > 333) {
      peaks++;
      lastPeak = tMs[i];
    }
  }
  final seconds = (tMs.last - tMs.first) / 1000;
  if (seconds <= 0) return null;
  final bpm = peaks * 60 / seconds;
  return (bpm < 35 || bpm > 200) ? null : bpm;
}
