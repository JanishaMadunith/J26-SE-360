import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:wellness_app/features/ppg_measure/ppg_math.dart';

/// Builds a YUV_420_888 frame. [uvPixelStride] 2 mimics semi-planar (interleaved) buffers.
({Uint8List y, Uint8List u, Uint8List v, int uvRowStride}) frame({
  required int width,
  required int height,
  required int Function(int x, int y) luma,
  required int u,
  required int v,
  int uvPixelStride = 1,
}) {
  final yPlane = Uint8List(width * height);
  for (var yy = 0; yy < height; yy++) {
    for (var xx = 0; xx < width; xx++) {
      yPlane[yy * width + xx] = luma(xx, yy);
    }
  }
  final uvRowStride = (width ~/ 2) * uvPixelStride;
  final uvSize = uvRowStride * (height ~/ 2);
  final uPlane = Uint8List(uvSize)..fillRange(0, uvSize, u);
  final vPlane = Uint8List(uvSize)..fillRange(0, uvSize, v);
  return (y: yPlane, u: uPlane, v: vPlane, uvRowStride: uvRowStride);
}

void main() {
  test('neutral chroma gives grey = luma', () {
    final f = frame(width: 64, height: 48, luma: (_, __) => 100, u: 128, v: 128);
    final m = yuv420Mean(
      width: 64,
      height: 48,
      yPlane: f.y,
      yRowStride: 64,
      uPlane: f.u,
      vPlane: f.v,
      uvRowStride: f.uvRowStride,
      uvPixelStride: 1,
    );
    expect(m.r, closeTo(100, 1e-9));
    expect(m.g, closeTo(100, 1e-9));
    expect(m.b, closeTo(100, 1e-9));
  });

  test('red-dominant chroma (finger over torch) and interleaved UV planes', () {
    final f = frame(width: 64, height: 48, luma: (_, __) => 100, u: 128, v: 200, uvPixelStride: 2);
    final m = yuv420Mean(
      width: 64,
      height: 48,
      yPlane: f.y,
      yRowStride: 64,
      uPlane: f.u,
      vPlane: f.v,
      uvRowStride: f.uvRowStride,
      uvPixelStride: 2,
    );
    expect(m.r, closeTo(100 + 1.402 * 72, 1e-6));
    expect(m.g, closeTo(100 - 0.714136 * 72, 1e-6));
    expect(m.b, closeTo(100, 1e-6));
    expect(m.r, greaterThan(m.g));
  });

  test('only the centre region is averaged', () {
    // Bright centre, black border: with a 50 % crop only the centre is sampled.
    final f = frame(
      width: 80,
      height: 80,
      luma: (x, y) => (x >= 20 && x < 60 && y >= 20 && y < 60) ? 200 : 0,
      u: 128,
      v: 128,
    );
    final m = yuv420Mean(
      width: 80,
      height: 80,
      yPlane: f.y,
      yRowStride: 80,
      uPlane: f.u,
      vPlane: f.v,
      uvRowStride: f.uvRowStride,
      uvPixelStride: 1,
    );
    expect(m.r, closeTo(200, 1e-9));
  });

  test('values are clipped to 0..255', () {
    final f = frame(width: 16, height: 16, luma: (_, __) => 250, u: 128, v: 255);
    final m = yuv420Mean(
      width: 16,
      height: 16,
      yPlane: f.y,
      yRowStride: 16,
      uPlane: f.u,
      vPlane: f.v,
      uvRowStride: f.uvRowStride,
      uvPixelStride: 1,
      step: 1,
    );
    expect(m.r, 255);
  });

  test('roughBpm finds a 72 bpm pulse in a clean 30 fps signal', () {
    const fps = 30;
    final tMs = <int>[];
    final red = <double>[];
    for (var i = 0; i < fps * 20; i++) {
      final t = i / fps;
      tMs.add((t * 1000).round());
      red.add(180 + 3 * math.sin(2 * math.pi * 1.2 * t) + 0.02 * i); // 1.2 Hz pulse + slow drift
    }
    final bpm = roughBpm(red, tMs);
    expect(bpm, isNotNull);
    expect(bpm!, closeTo(72, 6));
  });

  test('roughBpm refuses short recordings', () {
    expect(roughBpm(List.filled(30, 1.0), List.generate(30, (i) => i * 33)), isNull);
  });
}
