"""Knowledge Graph Schema: Entity Types, Relation Types, and Graph Data Models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeType(str, Enum):
    FOOD = "FOOD"
    NUTRIENT = "NUTRIENT"
    INGREDIENT = "INGREDIENT"
    ALLERGEN = "ALLERGEN"
    HEALTH_CONDITION = "HEALTH_CONDITION"
    DIETARY_PREFERENCE = "DIETARY_PREFERENCE"
    AFFECTIVE_STATE = "AFFECTIVE_STATE"
    COGNITIVE_STATE = "COGNITIVE_STATE"
    WELLNESS_TASK = "WELLNESS_TASK"
    MEAL_TYPE = "MEAL_TYPE"


class RelationType(str, Enum):
    # Food -> Nutrient / Ingredient
    CONTAINS = "CONTAINS"
    HAS_INGREDIENT = "HAS_INGREDIENT"

    # Nutrient/Task -> State
    ALLEVIATES = "ALLEVIATES"
    BOOSTS = "BOOSTS"
    REGULATES = "REGULATES"

    # Constraints / Safety
    CONTRAINDICATED_FOR = "CONTRAINDICATED_FOR"
    HAS_ALLERGEN = "HAS_ALLERGEN"
    COMPLIES_WITH = "COMPLIES_WITH"

    # Contextual
    SUITABLE_FOR_MEAL = "SUITABLE_FOR_MEAL"
    REQUIRES_TIME = "REQUIRES_TIME"


@dataclass(frozen=True)
class KGNode:
    id: str
    name: str
    node_type: NodeType
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KGTriple:
    head: str
    relation: RelationType
    tail: str
    weight: float = 1.0
    explanation: str = ""


@dataclass
class RawUserContext:
    """Raw user input snapshot for offline experimentation & evaluation."""
    user_id: str = "test-user-01"
    affective_state: str = "neutral"          # e.g. stress, anxiety, sad, happy, neutral, relaxed
    mental_fatigue: float = 0.2               # 0.0 to 1.0 (from M2 forecast)
    energy_level: float = 0.8                 # 0.0 to 1.0 (from M2 forecast)
    focus_level: float = 0.7                  # 0.0 to 1.0 (from M2 forecast)
    available_free_minutes: int = 45          # free slot from M3 schedule
    meal_type: str = "lunch"                  # breakfast, lunch, snack, dinner
    is_vegan: bool = False
    is_vegetarian: bool = False
    is_halal: bool = False
    allergies: list[str] = field(default_factory=list)          # e.g. ["nuts", "dairy", "fish"]
    dislikes: list[str] = field(default_factory=list)           # e.g. ["spicy", "mushrooms"]
    health_conditions: list[str] = field(default_factory=list)  # e.g. ["diabetes", "hypertension"]
    daily_calorie_target: int = 2000
    target_meal_calories: int = 600
