import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:wellness_app/data/api/apis.dart' show SendOutcome;
import 'package:wellness_app/sensing/upload_queue.dart';

import 'fixtures.dart';

void main() {
  late DateTime now;
  late InMemoryQueueStore store;
  late UploadQueue queue;

  setUp(() {
    now = DateTime.utc(2026, 10, 17, 9);
    store = InMemoryQueueStore();
    queue = UploadQueue(store, clock: () => now, maxAttempts: 3);
  });

  test('enqueue is idempotent on batch_id', () async {
    await queue.enqueue(sampleBatch());
    await queue.enqueue(sampleBatch());
    expect(await queue.pending(), 1);
  });

  test('payload is the exact SensorBatch JSON', () async {
    final batch = sampleBatch();
    await queue.enqueue(batch);
    final stored = store[batch.batchId]!;
    expect(jsonDecode(stored.payload), jsonDecode(jsonEncode(batch.toJson())));
  });

  test('sent and duplicate both remove the batch', () async {
    await queue.enqueue(sampleBatch());
    final r = await queue.flush((_) async => SendOutcome.duplicate);
    expect(r.sent, 1);
    expect(await queue.pending(), 0);
  });

  test('retry keeps the batch with exponential back-off and stops the flush', () async {
    await queue.enqueue(sampleBatch());
    var calls = 0;
    final r = await queue.flush((_) async {
      calls++;
      return SendOutcome.retry;
    });
    expect(r.retrying, 1);
    expect(calls, 1);
    final b = store[sampleBatch().batchId]!;
    expect(b.attempts, 1);
    expect(b.nextAttemptAt, now.add(UploadQueue.backoff(1)));
    // Not due yet → nothing is sent.
    expect((await queue.flush((_) async => SendOutcome.sent)).sent, 0);
    // After the back-off it goes through.
    now = now.add(const Duration(minutes: 1));
    expect((await queue.flush((_) async => SendOutcome.sent)).sent, 1);
  });

  test('a rejected payload is dropped so it cannot block the queue', () async {
    await queue.enqueue(sampleBatch());
    final r = await queue.flush((_) async => SendOutcome.rejected);
    expect(r.dropped, 1);
    expect(await queue.pending(), 0);
  });

  test('gives up after maxAttempts', () async {
    await queue.enqueue(sampleBatch());
    for (var i = 0; i < 3; i++) {
      await queue.flush((_) async => SendOutcome.retry);
      now = now.add(const Duration(hours: 7));
    }
    expect(await queue.pending(), 0);
  });

  test('back-off grows and is capped at 6 hours', () {
    expect(UploadQueue.backoff(1), const Duration(seconds: 30));
    expect(UploadQueue.backoff(2), const Duration(minutes: 1));
    expect(UploadQueue.backoff(30), const Duration(hours: 6));
  });
}
