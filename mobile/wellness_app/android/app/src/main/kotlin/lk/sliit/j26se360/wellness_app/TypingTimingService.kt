package lk.sliit.j26se360.wellness_app

import android.accessibilityservice.AccessibilityService
import android.content.ComponentName
import android.content.Context
import android.os.SystemClock
import android.provider.Settings
import android.view.accessibility.AccessibilityEvent
import org.json.JSONObject

/**
 * Records TYPING RHYTHM ONLY (Module 1 input).
 *
 * For every text change in any app it stores: when it happened and a key class
 * (letter / digit / space / backspace / enter / punctuation / other). The typed character is looked at
 * only long enough to classify it and is never stored, logged or sent. Password fields are ignored.
 * Accessibility events have no key-up, so hold_ms is always 0.
 */
class TypingTimingService : AccessibilityService() {

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        if (event == null || event.eventType != AccessibilityEvent.TYPE_VIEW_TEXT_CHANGED) return
        if (event.isPassword) return
        if (event.packageName?.toString() == packageName) return // our own app's text fields

        val keyClass = classify(event) ?: return
        // event.eventTime is on the uptime clock; convert to wall-clock milliseconds.
        val wallMs = System.currentTimeMillis() - (SystemClock.uptimeMillis() - event.eventTime)
        val record = JSONObject()
            .put("down_at_ms", wallMs)
            .put("hold_ms", 0)
            .put("key_class", keyClass)
        SensingFiles.append(this, SensingFiles.KEYSTROKES, record)
    }

    /** Returns the key class for a single-key change, or null for pastes / autocomplete / no-ops. */
    private fun classify(event: AccessibilityEvent): String? {
        val added = event.addedCount
        val removed = event.removedCount
        if (added == 0 && removed > 0) return "backspace"
        if (added != 1) return if (added > 1) "other" else null
        val text = event.text?.joinToString("") ?: return "other"
        val index = event.fromIndex
        if (index < 0 || index >= text.length) return "other"
        val c = text[index]
        return when {
            c == '\n' -> "enter"
            c.isWhitespace() -> "space"
            c.isLetter() -> "letter"
            c.isDigit() -> "digit"
            c in PUNCTUATION -> "punctuation"
            else -> "other"
        }
    }

    override fun onInterrupt() = Unit

    companion object {
        private const val PUNCTUATION = ".,;:!?'\"-()[]{}/\\@#&*_+=<>%$"

        fun isEnabled(context: Context): Boolean {
            val enabled = Settings.Secure.getString(
                context.contentResolver,
                Settings.Secure.ENABLED_ACCESSIBILITY_SERVICES,
            ) ?: return false
            val me = ComponentName(context, TypingTimingService::class.java)
            return enabled.split(':').any { ComponentName.unflattenFromString(it) == me }
        }
    }
}
