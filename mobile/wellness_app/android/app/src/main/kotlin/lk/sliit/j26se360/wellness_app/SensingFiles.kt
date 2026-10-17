package lk.sliit.j26se360.wellness_app

import android.content.Context
import org.json.JSONObject
import java.io.File

/**
 * JSON-lines buffers in filesDir/sensing shared with Dart (lib/sensing/file_bridge.dart).
 *
 * Writers append one JSON object per line. Readers claim a file by renaming it to
 * "<name>.draining-<nanos>" before reading, so a concurrent append always goes to a fresh file.
 */
object SensingFiles {
    const val KEYSTROKES = "keystrokes.jsonl"
    const val APP_USAGE = "app_usage.jsonl"
    const val SCREEN = "screen.jsonl"

    /** Hard cap per buffer so a long offline period cannot fill the phone. */
    private const val MAX_BYTES = 4L * 1024 * 1024

    private val lock = Any()

    fun dir(context: Context): File = File(context.filesDir, "sensing").apply { mkdirs() }

    /** Appends one record; returns false if the buffer is full (record dropped). */
    fun append(context: Context, name: String, record: JSONObject): Boolean = synchronized(lock) {
        val file = File(dir(context), name)
        if (file.exists() && file.length() > MAX_BYTES) return@synchronized false
        file.appendText(record.toString() + "\n")
        true
    }

    /** Returns and removes every buffered record (including leftovers of an interrupted drain). */
    fun drain(context: Context, name: String): List<JSONObject> = synchronized(lock) {
        val dir = dir(context)
        val current = File(dir, name)
        if (current.exists()) {
            current.renameTo(File(dir, "$name.draining-${System.nanoTime()}"))
        }
        val out = ArrayList<JSONObject>()
        val pending = dir.listFiles { f -> f.name.startsWith("$name.draining-") }?.sortedBy { it.name } ?: emptyList()
        for (f in pending) {
            f.forEachLine { line ->
                if (line.isNotBlank()) {
                    try {
                        out.add(JSONObject(line))
                    } catch (e: Exception) {
                        // skip a partially written line
                    }
                }
            }
            f.delete()
        }
        out
    }
}
