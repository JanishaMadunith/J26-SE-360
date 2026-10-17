"""User tasks (owned by the core API; scheduled by Module 3)."""

from enum import Enum
from uuid import UUID

from pydantic import AwareDatetime, Field, model_validator

from .common import Contract


class TaskCategory(str, Enum):
    study = "study"
    work = "work"
    exercise = "exercise"
    chores = "chores"
    personal = "personal"
    social = "social"
    other = "other"


class TaskStatus(str, Enum):
    pending = "pending"  # not yet placed in a schedule
    scheduled = "scheduled"  # has a slot in the current schedule
    done = "done"
    missed = "missed"
    cancelled = "cancelled"


class TaskCreate(Contract):
    title: str = Field(min_length=1, max_length=200)
    category: TaskCategory = TaskCategory.other
    estimated_minutes: int = Field(ge=5, le=12 * 60)
    deadline: AwareDatetime | None = None
    priority: int = Field(default=3, ge=1, le=5, description="1 = lowest, 5 = highest")
    fixed_start: AwareDatetime | None = Field(default=None, description="Set for fixed events (e.g. a lecture)")
    fixed_end: AwareDatetime | None = None
    notes: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def _fixed(self):
        if (self.fixed_start is None) != (self.fixed_end is None):
            raise ValueError("fixed_start and fixed_end must be given together")
        if self.fixed_start and self.fixed_end and self.fixed_end <= self.fixed_start:
            raise ValueError("fixed_end must be after fixed_start")
        return self


class TaskUpdate(Contract):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    category: TaskCategory | None = None
    estimated_minutes: int | None = Field(default=None, ge=5, le=12 * 60)
    deadline: AwareDatetime | None = None
    priority: int | None = Field(default=None, ge=1, le=5)
    status: TaskStatus | None = None
    notes: str | None = Field(default=None, max_length=1000)


class Task(TaskCreate):
    id: UUID
    status: TaskStatus
    created_at: AwareDatetime
    completed_at: AwareDatetime | None = None


class TaskEventType(str, Enum):
    """Every change is logged — this log is Module 3's future training data."""

    created = "created"
    scheduled = "scheduled"
    started = "started"
    completed = "completed"
    missed = "missed"
    rescheduled = "rescheduled"
    cancelled = "cancelled"
