package lk.sliit.j26se360.wellness_app

import android.content.Intent
import android.provider.Settings
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import org.json.JSONObject
import java.util.TimeZone

/**
 * Registers the sensing MethodChannel (lib/sensing/native_bridge.dart) and schedules the native
 * usage snapshot job. All heavy lifting lives in UsageCollector / TypingTimingService / SensingFiles.
 */
class MainActivity : FlutterActivity() {

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        UsageSnapshotJob.schedule(applicationContext)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL).setMethodCallHandler { call, result ->
            try {
                handle(call, result)
            } catch (e: Exception) {
                result.error("SENSING_ERROR", e.message, null)
            }
        }
    }

    private fun handle(call: MethodCall, result: MethodChannel.Result) {
        val ctx = applicationContext
        when (call.method) {
            "hasUsageAccess" -> result.success(UsageCollector.hasUsageAccess(ctx))
            "openUsageAccessSettings" -> {
                startActivity(Intent(Settings.ACTION_USAGE_ACCESS_SETTINGS).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
                result.success(null)
            }
            "getAppUsage" -> {
                val (startMs, endMs) = range(call)
                result.success(UsageCollector.appSessions(ctx, startMs, endMs).map { it.toMap() })
            }
            "getScreenEvents" -> {
                val (startMs, endMs) = range(call)
                result.success(UsageCollector.screenEvents(ctx, startMs, endMs).map { it.toMap() })
            }
            "isTypingServiceEnabled" -> result.success(TypingTimingService.isEnabled(ctx))
            "openAccessibilitySettings" -> {
                startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
                result.success(null)
            }
            "drainKeystrokes" -> result.success(SensingFiles.drain(ctx, SensingFiles.KEYSTROKES).map { toMap(it) })
            "snapshotNow" -> {
                // Off the main thread: queryEvents over a long gap can take a moment.
                Thread {
                    try {
                        UsageCollector.snapshot(ctx)
                        runOnUiThread { result.success(null) }
                    } catch (e: Exception) {
                        runOnUiThread { result.error("SENSING_ERROR", e.message, null) }
                    }
                }.start()
            }
            "getSensingDir" -> result.success(SensingFiles.dir(ctx).absolutePath)
            "getTimezone" -> result.success(TimeZone.getDefault().id)
            else -> result.notImplemented()
        }
    }

    private fun range(call: MethodCall): Pair<Long, Long> {
        val startMs = (call.argument<Any>("fromMs") as Number).toLong()
        val endMs = (call.argument<Any>("toMs") as Number).toLong()
        return Pair(startMs, endMs)
    }

    private fun toMap(json: JSONObject): Map<String, Any?> {
        val map = HashMap<String, Any?>()
        val keys = json.keys()
        while (keys.hasNext()) {
            val k = keys.next()
            map[k] = json.opt(k)
        }
        return map
    }

    companion object {
        const val CHANNEL = "lk.sliit.j26se360/sensing"
    }
}
