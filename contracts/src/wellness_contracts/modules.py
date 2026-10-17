"""Core API <-> module services. Data flows one way: M1 -> M2 -> M3 -> M4.

Ownership:
  M1 Affective State Detection ........ J.D. Jayatilake
  M2 Cognitive Wellness Prediction .... J.M. Samarathunga
  M3 Adaptive Schedule Optimization ... G.W.D.D.N. Kumarasinghe
  M4 Context-Aware Food Recommendation  T.C. Gunaweera
"""

from enum import Enum
from typing import Annotated
from uuid import UUID

from pydantic import AwareDatetime, Field

from .common import AffectiveLabel, Contract, ModelInfo, Score01
from .sensing import (
    ActivityTransition,
    AppUsageSession,
    KeystrokeEvent,
    LightReading,
    PPGRecording,
    ScreenEvent,
    SelfReport,
    StepCount,
)
from .tasks import TaskCategory


# ============================================================ Module 1 — Affective State Detection
class AffectiveStateRequest(Contract):
    """Raw signals for one analysis window (default: last 60 minutes)."""

    user_id: UUID
    window_start: AwareDatetime
    window_end: AwareDatetime
    timezone: str = "Asia/Colombo"
    ppg: list[PPGRecording] = []
    keystrokes: list[KeystrokeEvent] = []
    screen_events: list[ScreenEvent] = []
    app_usage: list[AppUsageSession] = []
    activity: list[ActivityTransition] = []
    steps: list[StepCount] = []
    light: list[LightReading] = []
    self_reports: list[SelfReport] = Field(default=[], description="Labels — for logging/training only, not inference")


class AffectiveStateResult(Contract):
    user_id: UUID
    window_start: AwareDatetime
    window_end: AwareDatetime
    label: AffectiveLabel
    probabilities: dict[AffectiveLabel, Score01]
    confidence: Score01
    features: dict[str, float | None] = Field(
        default={}, description="Features M1 extracted on the server, e.g. hr_bpm, hrv_rmssd_ms, typing_speed_kpm"
    )
    model: ModelInfo


# ============================================================ Module 2 — Cognitive Wellness Prediction
class WorkloadSummary(Contract):
    tasks_due_24h: int = 0
    minutes_due_24h: int = 0
    tasks_overdue: int = 0
    tasks_pending: int = 0


class WellnessForecastRequest(Contract):
    user_id: UUID
    as_of: AwareDatetime
    timezone: str = "Asia/Colombo"
    horizon_hours: int = Field(default=24, ge=1, le=72)
    slot_minutes: int = Field(
        default=60,
        ge=15,
        le=240,
        description="Forecast granularity. Module 3 needs intraday slots (resolves the daily-vs-slot issue).",
    )
    affective_history: list[AffectiveStateResult] = Field(default=[], description="Last 7 days of M1 outputs")
    screen_events: list[ScreenEvent] = Field(default=[], description="Last 48 h — for sleep estimation")
    activity: list[ActivityTransition] = []
    steps: list[StepCount] = []
    light: list[LightReading] = []
    self_reports: list[SelfReport] = []
    workload: WorkloadSummary = WorkloadSummary()


class WellnessSlot(Contract):
    start: AwareDatetime
    end: AwareDatetime
    mental_fatigue: Score01
    focus: Score01
    energy: Score01
    burnout_risk: Score01


class WellnessForecast(Contract):
    user_id: UUID
    generated_at: AwareDatetime
    slot_minutes: int
    slots: list[WellnessSlot]
    sleep_hours_last_night: float | None = None
    model: ModelInfo


# ============================================================ Module 3 — Predictive Adaptive Schedule Optimization
class SchedulableTask(Contract):
    task_id: UUID
    title: str
    category: TaskCategory
    estimated_minutes: int = Field(ge=5)
    priority: int = Field(ge=1, le=5)
    deadline: AwareDatetime | None = None
    fixed_start: AwareDatetime | None = None
    fixed_end: AwareDatetime | None = None
    times_rescheduled: int = 0


class BusyBlock(Contract):
    start: AwareDatetime
    end: AwareDatetime
    label: str = "busy"


class UserHistory(Contract):
    """Behaviour summary the completion model uses (computed by the core API from the task-event log)."""

    tasks_completed_30d: int = 0
    tasks_missed_30d: int = 0
    on_time_rate_30d: Score01 | None = None
    on_time_rate_by_category: dict[TaskCategory, Score01] = {}


class ScheduleRequest(Contract):
    user_id: UUID
    as_of: AwareDatetime
    timezone: str = "Asia/Colombo"
    horizon_start: AwareDatetime
    horizon_end: AwareDatetime
    day_start_hour: int = Field(default=7, ge=0, le=23, description="Earliest hour to place flexible tasks")
    day_end_hour: int = Field(default=22, ge=1, le=24)
    tasks: list[SchedulableTask]
    busy: list[BusyBlock] = []
    wellness: WellnessForecast | None = Field(default=None, description="If missing, M3 uses a neutral profile")
    history: UserHistory = UserHistory()
    reason: str = Field(default="scheduled", examples=["scheduled", "task_added", "overload_predicted", "task_missed"])


class ScheduledItem(Contract):
    task_id: UUID
    start: AwareDatetime
    end: AwareDatetime
    completion_probability: Score01
    is_fixed: bool = False


class UnscheduledTask(Contract):
    task_id: UUID
    reason: Annotated[str, Field(examples=["no_free_slot_before_deadline", "deadline_passed"])]


class ScheduleResult(Contract):
    user_id: UUID
    generated_at: AwareDatetime
    horizon_start: AwareDatetime
    horizon_end: AwareDatetime
    items: list[ScheduledItem]
    unscheduled: list[UnscheduledTask] = []
    at_risk_task_ids: list[UUID] = Field(default=[], description="Completion probability below the risk threshold")
    overload_predicted: bool = False
    reason: str
    model: ModelInfo


# ============================================================ Module 4 — Context-Aware Food Recommendation
class MealType(str, Enum):
    breakfast = "breakfast"
    lunch = "lunch"
    dinner = "dinner"
    snack = "snack"


class DietProfile(Contract):
    vegetarian: bool = False
    vegan: bool = False
    halal: bool = False
    allergies: list[str] = []
    dislikes: list[str] = []
    health_conditions: list[str] = Field(default=[], description="e.g. diabetes, hypertension — user-provided")
    daily_calorie_target: int | None = Field(default=None, ge=800, le=5000)


class RecommendationRequest(Contract):
    user_id: UUID
    as_of: AwareDatetime
    timezone: str = "Asia/Colombo"
    affective_state: AffectiveStateResult | None = None
    wellness: WellnessForecast | None = None
    schedule: ScheduleResult | None = None
    profile: DietProfile = DietProfile()


class FoodItem(Contract):
    name: str
    portion: str
    calories: int | None = None
    reason: str


class MealRecommendation(Contract):
    meal_type: MealType
    window_start: AwareDatetime
    window_end: AwareDatetime
    foods: list[FoodItem]
    rationale: str


class SuggestedActivity(Contract):
    """M4 'Part B'. Shown to the user only. If the user accepts, the app creates a normal task and M3
    schedules it on its next run — so data never flows back from M4 into M3 directly (no circular dependency)."""

    title: str
    category: TaskCategory
    estimated_minutes: int = Field(ge=5, le=240)
    reason: str


class Recommendations(Contract):
    user_id: UUID
    generated_at: AwareDatetime
    meals: list[MealRecommendation]
    suggested_activities: list[SuggestedActivity] = []
    model: ModelInfo


# ============================================================ pipeline status (core API)
class PipelineStep(str, Enum):
    affective = "affective"
    wellness = "wellness"
    schedule = "schedule"
    recommendations = "recommendations"


class StepStatus(Contract):
    step: PipelineStep
    ok: bool
    ms: int
    used_fallback: bool = False
    error: str | None = None
