"""Phone -> core API: raw sensor data.

All processing happens on the server (ADR-0004). To keep uploads small and respect privacy, the phone
sends compact RAW SIGNALS, never raw media or content:

* Camera PPG  -> per-frame mean R/G/B values (a 1-D signal, ~30 numbers/second), not video frames.
* Typing      -> key-press timings + key class only. The characters typed are NEVER captured.
* App usage   -> foreground sessions with app category and a salted hash of the package name.
* Screen      -> on / off / unlock events.
* Activity    -> activity-recognition transitions and step counts.
* Light       -> ambient light readings (used for sleep estimation).
* Self-report -> short mood/stress/energy/focus check-ins (ground-truth labels for Modules 1 and 2).
"""

from enum import Enum
from typing import Annotated
from uuid import UUID

from pydantic import AwareDatetime, Field, model_validator

from .common import Contract, Likert5


class PPGRecording(Contract):
    """One camera heart-rate measurement (finger over camera + flash). Arrays are column-wise and equal length."""

    started_at: AwareDatetime
    fps: float = Field(gt=0, le=240, description="Camera frame rate the values were sampled at")
    t_ms: list[int] = Field(description="Milliseconds since started_at for each frame")
    red: list[float] = Field(description="Mean red channel value per frame (0-255)")
    green: list[float]
    blue: list[float]
    torch_on: bool = True

    @model_validator(mode="after")
    def _equal_lengths(self):
        n = len(self.t_ms)
        if not (len(self.red) == len(self.green) == len(self.blue) == n):
            raise ValueError("t_ms, red, green and blue must have the same length")
        if n > 30 * 240:
            raise ValueError("recording too long: max 7,200 frames per recording")
        return self


class KeyClass(str, Enum):
    letter = "letter"
    digit = "digit"
    space = "space"
    backspace = "backspace"
    enter = "enter"
    punctuation = "punctuation"
    other = "other"


class KeystrokeEvent(Contract):
    """Timing of one key press. No character is ever recorded."""

    down_at: AwareDatetime
    hold_ms: int = Field(ge=0, le=10_000, description="Key-down to key-up time")
    key_class: KeyClass


class ScreenEventType(str, Enum):
    screen_on = "screen_on"
    screen_off = "screen_off"
    unlock = "unlock"


class ScreenEvent(Contract):
    at: AwareDatetime
    type: ScreenEventType


class AppCategory(str, Enum):
    """Android ApplicationInfo categories plus 'unknown'."""

    game = "game"
    audio = "audio"
    video = "video"
    image = "image"
    social = "social"
    news = "news"
    maps = "maps"
    productivity = "productivity"
    accessibility = "accessibility"
    unknown = "unknown"


class AppUsageSession(Contract):
    start: AwareDatetime
    end: AwareDatetime
    category: AppCategory
    app_hash: str = Field(min_length=8, max_length=64, description="Salted SHA-256 of the package name (prefix)")


class ActivityType(str, Enum):
    still = "still"
    walking = "walking"
    running = "running"
    on_bicycle = "on_bicycle"
    in_vehicle = "in_vehicle"
    unknown = "unknown"


class ActivityTransition(Contract):
    at: AwareDatetime
    activity: ActivityType
    entering: bool = Field(description="True when the activity starts, False when it ends")


class StepCount(Contract):
    start: AwareDatetime
    end: AwareDatetime
    steps: int = Field(ge=0)


class LightReading(Contract):
    at: AwareDatetime
    lux: float = Field(ge=0)


class SelfReport(Contract):
    """Short in-app check-in. Used as ground-truth labels when training Modules 1 and 2."""

    at: AwareDatetime
    mood: Likert5 | None = None
    stress: Likert5 | None = None
    energy: Likert5 | None = None
    focus: Likert5 | None = None
    note: Annotated[str, Field(max_length=280)] | None = None


class SensorBatch(Contract):
    """One upload from the phone. `batch_id` makes uploads idempotent (re-sending the same batch is safe)."""

    batch_id: UUID
    device_id: str = Field(min_length=4, max_length=64)
    app_version: str = Field(max_length=32)
    timezone: str = Field(examples=["Asia/Colombo"])
    collected_from: AwareDatetime
    collected_to: AwareDatetime
    ppg: list[PPGRecording] = []
    keystrokes: list[KeystrokeEvent] = []
    screen_events: list[ScreenEvent] = []
    app_usage: list[AppUsageSession] = []
    activity: list[ActivityTransition] = []
    steps: list[StepCount] = []
    light: list[LightReading] = []
    self_reports: list[SelfReport] = []

    @model_validator(mode="after")
    def _window(self):
        if self.collected_to < self.collected_from:
            raise ValueError("collected_to must be after collected_from")
        return self


class SensorBatchAck(Contract):
    batch_id: UUID
    accepted: bool
    duplicate: bool = False
    counts: dict[str, int] = {}
