import 'package:intl/intl.dart';

final _time = DateFormat.Hm();
final _dayTime = DateFormat('EEE d MMM, HH:mm');
final _day = DateFormat('EEE d MMM');

String formatTime(DateTime d) => _time.format(d.toLocal());

String formatDayTime(DateTime d) => _dayTime.format(d.toLocal());

String formatDay(DateTime d) => _day.format(d.toLocal());

String formatTimeRange(DateTime start, DateTime end) => '${formatTime(start)}–${formatTime(end)}';

String formatPercent(double v) => '${(v * 100).round()}%';

String formatMinutes(int minutes) {
  if (minutes < 60) return '$minutes min';
  final h = minutes ~/ 60;
  final m = minutes % 60;
  return m == 0 ? '$h h' : '$h h $m min';
}

String formatAgo(DateTime? d, {DateTime? now}) {
  if (d == null) return 'never';
  final diff = (now ?? DateTime.now()).difference(d);
  if (diff.inMinutes < 1) return 'just now';
  if (diff.inMinutes < 60) return '${diff.inMinutes} min ago';
  if (diff.inHours < 24) return '${diff.inHours} h ago';
  return formatDayTime(d);
}
