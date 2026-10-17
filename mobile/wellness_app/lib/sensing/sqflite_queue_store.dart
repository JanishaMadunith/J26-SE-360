import 'package:path/path.dart' as p;
import 'package:sqflite/sqflite.dart';

import 'upload_queue.dart';

/// Durable [QueueStore] backed by SQLite, so batches survive app restarts and reboots.
class SqfliteQueueStore implements QueueStore {
  SqfliteQueueStore._(this._db);

  static const _table = 'upload_queue';

  final Database _db;

  static Future<SqfliteQueueStore> open() async {
    final path = p.join(await getDatabasesPath(), 'wellness_upload_queue.db');
    final db = await openDatabase(
      path,
      version: 1,
      onCreate: (db, _) => db.execute('''
        CREATE TABLE $_table (
          id TEXT PRIMARY KEY,
          payload TEXT NOT NULL,
          attempts INTEGER NOT NULL,
          next_attempt_at INTEGER NOT NULL,
          created_at INTEGER NOT NULL
        )
      '''),
    );
    return SqfliteQueueStore._(db);
  }

  @override
  Future<void> insert(QueuedBatch batch) async {
    await _db.insert(_table, _toRow(batch), conflictAlgorithm: ConflictAlgorithm.ignore);
  }

  @override
  Future<List<QueuedBatch>> due(DateTime now, {int limit = 20}) async {
    final rows = await _db.query(
      _table,
      where: 'next_attempt_at <= ?',
      whereArgs: [now.millisecondsSinceEpoch],
      orderBy: 'created_at ASC',
      limit: limit,
    );
    return rows.map(_fromRow).toList();
  }

  @override
  Future<void> update(QueuedBatch batch) async {
    await _db.update(_table, _toRow(batch), where: 'id = ?', whereArgs: [batch.id]);
  }

  @override
  Future<void> remove(String id) async {
    await _db.delete(_table, where: 'id = ?', whereArgs: [id]);
  }

  @override
  Future<int> count() async {
    final rows = await _db.rawQuery('SELECT COUNT(*) AS n FROM $_table');
    return (rows.first['n'] as num?)?.toInt() ?? 0;
  }

  static Map<String, Object?> _toRow(QueuedBatch b) => {
        'id': b.id,
        'payload': b.payload,
        'attempts': b.attempts,
        'next_attempt_at': b.nextAttemptAt.millisecondsSinceEpoch,
        'created_at': b.createdAt.millisecondsSinceEpoch,
      };

  static QueuedBatch _fromRow(Map<String, Object?> r) => QueuedBatch(
        id: r['id']! as String,
        payload: r['payload']! as String,
        attempts: (r['attempts']! as num).toInt(),
        nextAttemptAt: DateTime.fromMillisecondsSinceEpoch((r['next_attempt_at']! as num).toInt()),
        createdAt: DateTime.fromMillisecondsSinceEpoch((r['created_at']! as num).toInt()),
      );
}
