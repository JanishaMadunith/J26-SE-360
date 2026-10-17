"""Types shared by every contract."""

from datetime import UTC, datetime
from enum import Enum
from typing import Annotated

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

Score01 = Annotated[float, Field(ge=0.0, le=1.0)]
"""A normalised score: 0 = none/lowest, 1 = maximum."""

Likert5 = Annotated[int, Field(ge=1, le=5)]


class Contract(BaseModel):
    """Base for all contracts: unknown fields are rejected so mistakes surface immediately."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True, use_enum_values=False)


class ModelInfo(Contract):
    """Which model produced an output — stored with every result so experiments are reproducible."""

    name: str = Field(examples=["m3-heuristic"])
    version: str = Field(examples=["0.1.0"])


class TimeWindow(Contract):
    start: AwareDatetime
    end: AwareDatetime


class AffectiveLabel(str, Enum):
    """Affective states from the Topic Assessment Form (section 7).
    OPEN ISSUE: some team documents use three classes — agree the final set, then update here only."""

    stress = "stress"
    relaxed = "relaxed"
    happy = "happy"
    sad = "sad"
    anxiety = "anxiety"


def utc_now() -> datetime:
    return datetime.now(UTC)
