import 'dart:async';
import 'dart:math' as math;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:permission_handler/permission_handler.dart';

import '../../data/models/sensing.dart';
import '../../sensing/sensing_providers.dart';
import 'ppg_math.dart';

enum _Phase { idle, starting, measuring, uploading, done, error }

/// Camera photoplethysmography: finger over the back camera with the torch on for 60 s.
/// Only per-frame mean R/G/B values are kept and uploaded (Module 1 computes HR/HRV on the server).
class PpgScreen extends ConsumerStatefulWidget {
  const PpgScreen({super.key});

  @override
  ConsumerState<PpgScreen> createState() => _PpgScreenState();
}

class _PpgScreenState extends ConsumerState<PpgScreen> {
  static const _duration = Duration(seconds: 60);

  CameraController? _controller;
  _Phase _phase = _Phase.idle;
  String? _message;
  final _stopwatch = Stopwatch();
  DateTime? _startedAt;
  final _tMs = <int>[];
  final _red = <double>[];
  final _green = <double>[];
  final _blue = <double>[];
  Timer? _ticker;

  @override
  void dispose() {
    _ticker?.cancel();
    final c = _controller;
    if (c != null) {
      if (c.value.isStreamingImages) unawaited(c.stopImageStream());
      unawaited(c.dispose());
    }
    super.dispose();
  }

  Future<void> _start() async {
    setState(() {
      _phase = _Phase.starting;
      _message = null;
    });
    try {
      if (!(await Permission.camera.request()).isGranted) {
        throw StateError('Camera permission is needed to measure heart rate.');
      }
      final cameras = await availableCameras();
      final back = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );
      final controller = CameraController(
        back,
        ResolutionPreset.low,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );
      await controller.initialize();
      await controller.setFlashMode(FlashMode.torch);
      _controller = controller;
      _tMs.clear();
      _red.clear();
      _green.clear();
      _blue.clear();
      _startedAt = DateTime.now();
      _stopwatch
        ..reset()
        ..start();
      await controller.startImageStream(_onFrame);
      _ticker = Timer.periodic(const Duration(milliseconds: 250), (_) {
        if (_stopwatch.elapsed >= _duration) {
          unawaited(_finish());
        } else if (mounted) {
          setState(() {});
        }
      });
      if (mounted) setState(() => _phase = _Phase.measuring);
    } catch (e) {
      await _stopCamera();
      if (mounted) {
        setState(() {
          _phase = _Phase.error;
          _message = '$e';
        });
      }
    }
  }

  void _onFrame(CameraImage image) {
    if (_phase != _Phase.measuring && _phase != _Phase.starting) return;
    if (_tMs.length >= PPGRecording.maxFrames || image.planes.length < 3) return;
    final y = image.planes[0];
    final u = image.planes[1];
    final v = image.planes[2];
    final mean = yuv420Mean(
      width: image.width,
      height: image.height,
      yPlane: y.bytes,
      yRowStride: y.bytesPerRow,
      uPlane: u.bytes,
      vPlane: v.bytes,
      uvRowStride: u.bytesPerRow,
      uvPixelStride: u.bytesPerPixel ?? 1,
    );
    _tMs.add(_stopwatch.elapsedMilliseconds);
    _red.add(mean.r);
    _green.add(mean.g);
    _blue.add(mean.b);
  }

  Future<void> _stopCamera() async {
    _ticker?.cancel();
    _ticker = null;
    _stopwatch.stop();
    final c = _controller;
    _controller = null;
    if (c == null) return;
    try {
      if (c.value.isStreamingImages) await c.stopImageStream();
      await c.setFlashMode(FlashMode.off);
    } catch (_) {}
    await c.dispose();
  }

  Future<void> _finish() async {
    if (_phase != _Phase.measuring) return;
    setState(() => _phase = _Phase.uploading);
    await _stopCamera();
    try {
      if (_tMs.length < 300) throw StateError('Too few frames captured (${_tMs.length}). Keep your finger still and try again.');
      final seconds = _tMs.last / 1000;
      final recording = PPGRecording(
        startedAt: _startedAt!,
        fps: math.min(240.0, _tMs.length / seconds),
        tMs: List.of(_tMs),
        red: _red.map(_round2).toList(),
        green: _green.map(_round2).toList(),
        blue: _blue.map(_round2).toList(),
      );
      final repo = await ref.read(sensingRepositoryProvider.future);
      await repo.enqueue(repo.singleBatch(ppg: [recording]));
      final result = await repo.flush();
      if (!mounted) return;
      setState(() {
        _phase = _Phase.done;
        _message = result.sent > 0
            ? 'Measurement uploaded (${_tMs.length} frames). Results appear after the next analysis.'
            : 'Saved — it will upload when the server is reachable.';
      });
    } catch (e) {
      if (mounted) {
        setState(() {
          _phase = _Phase.error;
          _message = '$e';
        });
      }
    }
  }

  static double _round2(double v) => (v * 100).roundToDouble() / 100;

  Future<void> _cancel() async {
    await _stopCamera();
    if (mounted) setState(() => _phase = _Phase.idle);
  }

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    final scheme = Theme.of(context).colorScheme;
    final remaining = _duration - _stopwatch.elapsed;
    final bpm = _phase == _Phase.measuring ? roughBpm(_red, _tMs) : null;
    final redLooksCovered = _red.isNotEmpty && _red.last > 120 && _green.last < 100;

    return Scaffold(
      appBar: AppBar(title: const Text('Measure heart rate')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text(
              'Cover the back camera and flash completely with your fingertip. Keep still and breathe '
              'normally for 60 seconds.',
              style: text.bodyMedium,
            ),
            const SizedBox(height: 16),
            if (_controller != null && _controller!.value.isInitialized)
              SizedBox(height: 120, child: Center(child: AspectRatio(aspectRatio: 1, child: CameraPreview(_controller!))))
            else
              const SizedBox(height: 120, child: Icon(Icons.fingerprint, size: 96)),
            const SizedBox(height: 16),
            SizedBox(
              height: 120,
              child: CustomPaint(painter: SignalPainter(_red, color: scheme.primary)),
            ),
            const SizedBox(height: 16),
            if (_phase == _Phase.measuring) ...[
              LinearProgressIndicator(
                value: (1 - remaining.inMilliseconds / _duration.inMilliseconds).clamp(0.0, 1.0).toDouble(),
              ),
              const SizedBox(height: 8),
              Text('${remaining.inSeconds.clamp(0, 60)} s left · ${_tMs.length} frames', textAlign: TextAlign.center),
              Text(
                redLooksCovered ? (bpm == null ? 'Reading…' : '≈ ${bpm.round()} bpm (preview)') : 'Finger not detected — cover the lens',
                textAlign: TextAlign.center,
                style: text.titleMedium?.copyWith(color: redLooksCovered ? scheme.primary : scheme.error),
              ),
            ],
            if (_message != null)
              Padding(
                padding: const EdgeInsets.only(top: 12),
                child: Text(
                  _message!,
                  textAlign: TextAlign.center,
                  style: TextStyle(color: _phase == _Phase.error ? scheme.error : null),
                ),
              ),
            const Spacer(),
            switch (_phase) {
              _Phase.idle || _Phase.done || _Phase.error =>
                FilledButton.icon(onPressed: _start, icon: const Icon(Icons.favorite), label: const Text('Start')),
              _Phase.starting || _Phase.uploading => const Center(child: CircularProgressIndicator()),
              _Phase.measuring => OutlinedButton(onPressed: _cancel, child: const Text('Cancel')),
            },
          ],
        ),
      ),
    );
  }
}

/// Draws the last ~6 seconds of the red channel, auto-scaled.
class SignalPainter extends CustomPainter {
  SignalPainter(this.values, {required this.color}) : _length = values.length;

  final List<double> values;
  final Color color;
  final int _length;

  @override
  void paint(Canvas canvas, Size size) {
    if (values.length < 2) return;
    final start = math.max(0, values.length - 180);
    final window = values.sublist(start);
    final lo = window.reduce(math.min);
    final hi = window.reduce(math.max);
    final range = (hi - lo).abs() < 1e-6 ? 1.0 : hi - lo;
    final path = Path();
    for (var i = 0; i < window.length; i++) {
      final x = size.width * i / (window.length - 1);
      final y = size.height - (window[i] - lo) / range * size.height;
      if (i == 0) {
        path.moveTo(x, y);
      } else {
        path.lineTo(x, y);
      }
    }
    canvas.drawPath(
      path,
      Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2,
    );
  }

  @override
  bool shouldRepaint(SignalPainter oldDelegate) => oldDelegate._length != _length || oldDelegate.color != color;
}
