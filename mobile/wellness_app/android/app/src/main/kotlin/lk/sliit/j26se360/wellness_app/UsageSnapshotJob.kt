package lk.sliit.j26se360.wellness_app

import android.app.job.JobInfo
import android.app.job.JobParameters
import android.app.job.JobScheduler
import android.app.job.JobService
import android.content.ComponentName
import android.content.Context

/**
 * Every ~15 minutes (Android minimum) writes new app-usage and screen events into the JSONL buffers,
 * so the Flutter background worker — which cannot call the MethodChannel — can upload them.
 * Uses the framework JobScheduler, so no extra Gradle dependency is needed. Survives reboots (persisted).
 */
class UsageSnapshotJob : JobService() {

    override fun onStartJob(params: JobParameters?): Boolean {
        Thread {
            try {
                UsageCollector.snapshot(applicationContext)
            } catch (e: Exception) {
                // Never crash the app from a background snapshot.
            } finally {
                jobFinished(params, false)
            }
        }.start()
        return true // work continues on the thread above
    }

    override fun onStopJob(params: JobParameters?): Boolean = true // reschedule if stopped early

    companion object {
        private const val JOB_ID = 26_0360

        fun schedule(context: Context) {
            val scheduler = context.getSystemService(Context.JOB_SCHEDULER_SERVICE) as JobScheduler
            if (scheduler.getPendingJob(JOB_ID) != null) return
            val job = JobInfo.Builder(JOB_ID, ComponentName(context, UsageSnapshotJob::class.java))
                .setPeriodic(15 * 60 * 1000L)
                .setPersisted(true) // requires RECEIVE_BOOT_COMPLETED
                .build()
            scheduler.schedule(job)
        }
    }
}
