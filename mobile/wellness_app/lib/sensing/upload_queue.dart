import 'dart:convert';
import 'dart:math' as math;

import '../data/api/apis.dart' show SendOutcome;
import '../data/models/sensing.dart';

/// One batch waiting to be uploaded. [id] is the batch_id, so inserting the same batch twice is a no-op
/// and re-sending after a lost response is safe (the server answers `duplicate`).
class QueuedBatch {
  const QueuedBatch({
    required this.id,
    required this.payload,
    required this.attempts,
    required this.nextAttemptAt,
    required this.createdAt,
  });

  final String id;
  final String payload;
  final int attempts;
  final DateTime nextAttemptAt;
  final DateTime createdAt;

  QueuedBatch retryLater(DateTime now) => QueuedBatch(
        id: id,
        payload: payload,
        attempts: attempts + 1,
        nextAttemptAt: now.add(UploadQueue.backoff(attempts + 1)),
        createdAt: createdAt,
      );
}

/// Persistence behind the queue (sqflite in the app, in-memory in tests).
abstract interface class QueueStore {
  /// Must ignore an insert whose id already exists.
  Future<void> insert(QueuedBatch batch);
  Future<List<QueuedBatch>> due(DateTime now, {int limit = 20});
  Future<void> update(QueuedBatch batch);
  Future<void> remove(String id);
  Future<int> count();
}

class FlushResult {
  const FlushResult({this.sent = 0, this.retrying = 0, this.dropped = 0});

  final int sent;
  final int retrying;
  final int dropped;

  @override
  String toString() => 'sent $sent, retrying $retrying, dropped $dropped';
}

/// Offline-first upload queue with exponential back-off.
class UploadQueue {
  UploadQueue(this.store, {this.maxAttempts = 40, DateTime Function()? clock}) : _clock = clock ?? DateTime.now;

  final QueueStore store;
  final int maxAttempts;
  final DateTime Function() _clock;

  /// 30 s, 1 min, 2 min, … capped at 6 hours.
  static Duration backoff(int attempts) {
    final seconds = 30 * math.pow(2, math.max(0, attempts - 1));
    return Duration(seconds: math.min(seconds, 6 * 3600).toInt());
  }

  Future<void> enqueue(SensorBatch batch) {
    final now = _clock();
    return store.insert(
      QueuedBatch(id: batch.batchId, payload: jsonEncode(batch.toJson()), attempts: 0, nextAttemptAt: now, createdAt: now),
    );
  }

  Future<int> pending() => store.count();

  /// Sends every due batch in order. [send] must not throw.
  Future<FlushResult> flush(Future<SendOutcome> Function(String payload) send, {int limit = 20}) async {
    var sent = 0;
    var retrying = 0;
    var dropped = 0;
    for (final b in await store.due(_clock(), limit: limit)) {
      final outcome = await send(b.payload);
      switch (outcome) {
        case SendOutcome.sent:
        case SendOutcome.duplicate:
          await store.remove(b.id);
          sent++;
        case SendOutcome.rejected:
          // The server will never accept this payload (validation error); keep the queue moving.
          await store.remove(b.id);
          dropped++;
        case SendOutcome.retry:
          if (b.attempts + 1 >= maxAttempts) {
            await store.remove(b.id);
            dropped++;
          } else {
            await store.update(b.retryLater(_clock()));
            retrying++;
          }
      }
      if (outcome == SendOutcome.retry) break; // server/network down: stop and try again later
    }
    return FlushResult(sent: sent, retrying: retrying, dropped: dropped);
  }
}

/// In-memory [QueueStore] for tests and previews.
class InMemoryQueueStore implements QueueStore {
  final Map<String, QueuedBatch> _items = {};

  @override
  Future<void> insert(QueuedBatch batch) async {
    _items.putIfAbsent(batch.id, () => batch);
  }

  @override
  Future<List<QueuedBatch>> due(DateTime now, {int limit = 20}) async {
    final list = _items.values.where((b) => !b.nextAttemptAt.isAfter(now)).toList()
      ..sort((a, b) => a.createdAt.compareTo(b.createdAt));
    return list.take(limit).toList();
  }

  @override
  Future<void> update(QueuedBatch batch) async {
    _items[batch.id] = batch;
  }

  @override
  Future<void> remove(String id) async {
    _items.remove(id);
  }

  @override
  Future<int> count() async => _items.length;

  QueuedBatch? operator [](String id) => _items[id];
}
