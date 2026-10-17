package lk.sliit.j26se360.wellness_app

import android.app.AppOpsManager
import android.app.usage.UsageEvents
import android.app.usage.UsageStatsManager
import android.content.Context
import android.content.pm.ApplicationInfo
import android.content.pm.PackageManager
import android.os.Build
import android.os.Process
import org.json.JSONObject
import java.security.MessageDigest
import java.util.UUID

/**
 * Reads app-usage and screen events from UsageStatsManager (needs the "Usage access" special permission).
 *
 * Privacy: package names never leave this class — they are replaced by a salted SHA-256 prefix
 * (per-install salt) plus the app's Play Store category.
 */
object UsageCollector {
    // UsageEvents.Event types (numeric so they compile against minSdk 26; names in comments).
    private const val ACTIVITY_RESUMED = 1        // MOVE_TO_FOREGROUND (API <29 name)
    private const val ACTIVITY_PAUSED = 2         // MOVE_TO_BACKGROUND
    private const val SCREEN_INTERACTIVE = 15     // API 28+
    private const val SCREEN_NON_INTERACTIVE = 16 // API 28+
    private const val KEYGUARD_HIDDEN = 18        // API 28+ (user unlocked)

    private const val PREFS = "wellness_sensing"
    private const val KEY_SALT = "salt"
    private const val KEY_CURSOR = "usage_cursor_ms"
    private const val MIN_SESSION_MS = 1_000L

    data class AppSession(val startMs: Long, val endMs: Long, val category: String, val appHash: String) {
        fun toMap(): Map<String, Any> = mapOf("start_ms" to startMs, "end_ms" to endMs, "category" to category, "app_hash" to appHash)
        fun toJson(): JSONObject = JSONObject(toMap())
    }

    data class ScreenEvent(val atMs: Long, val type: String) {
        fun toMap(): Map<String, Any> = mapOf("at_ms" to atMs, "type" to type)
        fun toJson(): JSONObject = JSONObject(toMap())
    }

    fun hasUsageAccess(context: Context): Boolean {
        val appOps = context.getSystemService(Context.APP_OPS_SERVICE) as AppOpsManager
        val mode = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            appOps.unsafeCheckOpNoThrow(AppOpsManager.OPSTR_GET_USAGE_STATS, Process.myUid(), context.packageName)
        } else {
            legacyCheckOp(appOps, context)
        }
        return mode == AppOpsManager.MODE_ALLOWED
    }

    @Suppress("DEPRECATION")
    private fun legacyCheckOp(appOps: AppOpsManager, context: Context): Int =
        appOps.checkOpNoThrow(AppOpsManager.OPSTR_GET_USAGE_STATS, Process.myUid(), context.packageName)

    /** Foreground app sessions overlapping [fromMs, toMs). */
    fun appSessions(context: Context, fromMs: Long, toMs: Long): List<AppSession> {
        if (!hasUsageAccess(context) || toMs <= fromMs) return emptyList()
        val usm = context.getSystemService(Context.USAGE_STATS_SERVICE) as UsageStatsManager
        val events = usm.queryEvents(fromMs, toMs)
        val event = UsageEvents.Event()
        val open = HashMap<String, Long>()
        val sessions = ArrayList<AppSession>()
        val categoryCache = HashMap<String, String>()

        fun close(pkg: String, start: Long, end: Long) {
            if (end - start >= MIN_SESSION_MS) {
                val category = categoryCache.getOrPut(pkg) { categoryOf(context, pkg) }
                sessions.add(AppSession(start, end, category, hash(context, pkg)))
            }
        }

        while (events.hasNextEvent()) {
            events.getNextEvent(event)
            val pkg = event.packageName ?: continue
            when (event.eventType) {
                ACTIVITY_RESUMED -> {
                    // A new foreground app ends any other open session.
                    for ((other, start) in open.entries.toList()) {
                        if (other != pkg) {
                            close(other, start, event.timeStamp)
                            open.remove(other)
                        }
                    }
                    if (!open.containsKey(pkg)) open[pkg] = event.timeStamp
                }
                ACTIVITY_PAUSED -> {
                    // Unmatched pause = the app was already in front when the window began.
                    val start = open.remove(pkg) ?: fromMs
                    close(pkg, start, event.timeStamp)
                }
            }
        }
        for ((pkg, start) in open) close(pkg, start, toMs)
        return sessions.sortedBy { it.startMs }
    }

    /** Screen on / off / unlock events in [fromMs, toMs) (Android 9+; empty on older versions). */
    fun screenEvents(context: Context, fromMs: Long, toMs: Long): List<ScreenEvent> {
        if (!hasUsageAccess(context) || toMs <= fromMs || Build.VERSION.SDK_INT < Build.VERSION_CODES.P) return emptyList()
        val usm = context.getSystemService(Context.USAGE_STATS_SERVICE) as UsageStatsManager
        val events = usm.queryEvents(fromMs, toMs)
        val event = UsageEvents.Event()
        val out = ArrayList<ScreenEvent>()
        while (events.hasNextEvent()) {
            events.getNextEvent(event)
            val type = when (event.eventType) {
                SCREEN_INTERACTIVE -> "screen_on"
                SCREEN_NON_INTERACTIVE -> "screen_off"
                KEYGUARD_HIDDEN -> "unlock"
                else -> null
            } ?: continue
            out.add(ScreenEvent(event.timeStamp, type))
        }
        return out
    }

    /**
     * Writes everything since the stored cursor into the JSONL buffers and advances the cursor.
     * Called by UsageSnapshotJob every ~15 minutes and by MainActivity on demand ("snapshotNow").
     */
    @Synchronized
    fun snapshot(context: Context) {
        if (!hasUsageAccess(context)) return
        val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val now = System.currentTimeMillis()
        val from = prefs.getLong(KEY_CURSOR, now - 60 * 60 * 1000L)
        if (now - from < 1_000L) return
        for (s in appSessions(context, from, now)) SensingFiles.append(context, SensingFiles.APP_USAGE, s.toJson())
        for (e in screenEvents(context, from, now)) SensingFiles.append(context, SensingFiles.SCREEN, e.toJson())
        prefs.edit().putLong(KEY_CURSOR, now).apply()
    }

    private fun categoryOf(context: Context, pkg: String): String = try {
        @Suppress("DEPRECATION")
        val info = context.packageManager.getApplicationInfo(pkg, 0)
        when (info.category) {
            ApplicationInfo.CATEGORY_GAME -> "game"
            ApplicationInfo.CATEGORY_AUDIO -> "audio"
            ApplicationInfo.CATEGORY_VIDEO -> "video"
            ApplicationInfo.CATEGORY_IMAGE -> "image"
            ApplicationInfo.CATEGORY_SOCIAL -> "social"
            ApplicationInfo.CATEGORY_NEWS -> "news"
            ApplicationInfo.CATEGORY_MAPS -> "maps"
            ApplicationInfo.CATEGORY_PRODUCTIVITY -> "productivity"
            8 -> "accessibility" // ApplicationInfo.CATEGORY_ACCESSIBILITY (API 31)
            else -> "unknown"
        }
    } catch (e: PackageManager.NameNotFoundException) {
        "unknown"
    }

    private fun salt(context: Context): String {
        val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        prefs.getString(KEY_SALT, null)?.let { return it }
        val fresh = UUID.randomUUID().toString()
        prefs.edit().putString(KEY_SALT, fresh).apply()
        return fresh
    }

    /** First 16 hex chars of SHA-256(salt + ":" + packageName). */
    fun hash(context: Context, pkg: String): String {
        val digest = MessageDigest.getInstance("SHA-256").digest("${salt(context)}:$pkg".toByteArray(Charsets.UTF_8))
        return digest.joinToString("") { "%02x".format(it) }.substring(0, 16)
    }
}
