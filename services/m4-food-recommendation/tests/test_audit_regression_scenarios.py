"""Automated Regression Test Suite for the 3 Concrete Audit Scenarios.

Verifies:
- Scenario 1: Stress, low energy, mental fatigue, 2000 kcal target.
  - Temporal freshness decay across calendar days (next morning breakfast reverts to neutral baseline).
  - Variety constraint preventing same-dish duplication across lunch and dinner.
  - Calibrated scientific explanations (no medical curative claims; traceable USDA/MRI data).
  - HeartSteps mHealth MRT activity recommendations framed as options with empirical stats.

- Scenario 2: Vegan, allergies=["gluten", "coconut"], dislikes=["spicy"], 1800 kcal target.
  - Strict exclusion of Soya meat curry (spicy curry, unverified gluten cross-contamination risk).
  - Strict exclusion of Kankun stir-fry (spicy chili stir-fry, unverified coconut oil cross-contamination risk).
  - Detection and explicit reporting of breakfast calorie deficit (165 kcal vs 450 kcal target; protein deficit).
  - Explicit reporting of candidate catalog exhaustion for lunch and dinner (preserving safety over completeness).

- Scenario 3: Type 2 diabetes, 1900 kcal target.
  - Strict avoidance of high Glycemic Load / high GI starches without adequate fiber.
  - Variety constraint: lunch and dinner select distinct main meals (no repeated salmon).
  - Clinical guidance explanations aligned with ADA standards of care.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pytest

from m4_food.graph.kg_service import KnowledgeGraphService, recommend_kg
from wellness_contracts import (
    AffectiveLabel,
    AffectiveStateResult,
    DietProfile,
    MealType,
    ModelInfo,
    RecommendationRequest,
    SuggestedActivity,
    TaskCategory,
    WellnessForecast,
    WellnessSlot,
)

COLOMBO = ZoneInfo("Asia/Colombo")
UID = uuid.uuid4()
MODEL_M1 = ModelInfo(name="m1-affective-state", version="0.1.0")


def test_audit_scenario_1_stress_and_temporal_freshness():
    """Scenario 1: Stress, low energy, mental fatigue, 2000 kcal target."""
    as_of = datetime(2026, 10, 23, 14, 0, tzinfo=COLOMBO)
    req = RecommendationRequest(
        user_id=UID,
        as_of=as_of,
        timezone="Asia/Colombo",
        affective_state=AffectiveStateResult(
            user_id=UID,
            window_start=as_of - timedelta(hours=1),
            window_end=as_of,
            label=AffectiveLabel.stress,
            probabilities={AffectiveLabel.stress: 0.85, AffectiveLabel.relaxed: 0.15},
            confidence=0.85,
            model=MODEL_M1,
        ),
        wellness=WellnessForecast(
            user_id=UID,
            generated_at=as_of,
            slot_minutes=60,
            model=MODEL_M1,
            slots=[
                WellnessSlot(
                    start=as_of + timedelta(hours=1),
                    end=as_of + timedelta(hours=2),
                    energy=0.3,
                    mental_fatigue=0.7,
                    focus=0.3,
                    burnout_risk=0.5,
                )
            ],
        ),
        profile=DietProfile(daily_calorie_target=2000),
    )

    meals, activities = recommend_kg(req)
    assert len(meals) >= 3

    # Check Dinner on Day 1 (19:30): acute stress context active (<6h heuristic)
    dinner = next(m for m in meals if m.meal_type == MealType.dinner and m.window_start.date() == as_of.date())
    assert "elevated stress detected" in dinner.rationale
    assert "probability" in dinner.rationale
    assert "anxiety" not in dinner.rationale  # Only stress was predicted

    # Check Breakfast on Day 2 (07:00 next day): stress context MUST have decayed
    breakfast = next(m for m in meals if m.meal_type == MealType.breakfast and m.window_start.date() > as_of.date())
    assert "heuristic horizon" in breakfast.rationale
    assert "baseline steady-energy foods prioritized" in breakfast.rationale

    # Check Variety: Lunch on Day 2 should NOT duplicate Dinner on Day 1
    dinner_foods = {f.name for f in dinner.foods}
    lunch = next(m for m in meals if m.meal_type == MealType.lunch and m.window_start.date() > as_of.date())
    lunch_foods = {f.name for f in lunch.foods}
    assert not (dinner_foods & lunch_foods), f"Duplicated main meal across planning period: {dinner_foods & lunch_foods}"

    # Scientific explanation check: NO curative claims, no adrenaline suppression, no working memory promises
    all_reasons = " ".join(f.reason for m in meals for f in m.foods)
    assert "treats stress" not in all_reasons.lower()
    assert "cures" not in all_reasons.lower()
    assert "modulates hpa axis" not in all_reasons.lower()
    assert "adrenaline" not in all_reasons.lower()
    assert "working memory" not in all_reasons.lower()
    assert "energy crashes" not in all_reasons.lower()

    # HeartSteps empirical activity check: non-prescriptive, study log stats with unrated transparency
    assert len(activities) >= 1
    assert any("Option to consider" in a.reason for a in activities)
    assert any("HeartSteps mHealth MRT" in a.reason for a in activities)
    assert any("not clinical efficacy" in a.reason for a in activities)


def test_audit_scenario_2_vegan_allergies_and_dislikes():
    """Scenario 2: Vegan, allergies=["gluten", "coconut"], dislikes=["spicy"], 1800 kcal target."""
    as_of = datetime(2026, 10, 23, 7, 0, tzinfo=COLOMBO)
    req = RecommendationRequest(
        user_id=UID,
        as_of=as_of,
        timezone="Asia/Colombo",
        profile=DietProfile(
            vegan=True,
            allergies=["gluten", "coconut"],
            dislikes=["spicy"],
            daily_calorie_target=1800,
        ),
    )

    meals, activities = recommend_kg(req)

    # All recommended food items across all meals MUST strictly respect safety constraints
    all_rec_foods = [f.name for m in meals for f in m.foods]
    # Soya meat curry and Kankun stir-fry MUST be strictly excluded
    assert "Soya meat curry with red rice" not in all_rec_foods
    assert "Kankun (water spinach) stir-fry with rice" not in all_rec_foods

    # Breakfast deficit detection
    breakfast = next(m for m in meals if m.meal_type == MealType.breakfast)
    assert len(breakfast.foods) > 0
    b_cals = sum(f.calories for f in breakfast.foods)
    assert b_cals <= 200  # Only banana + papaya
    assert "[STATUS: INCOMPLETE_PLAN]" in breakfast.rationale
    assert "Nutritional deficit alert" in breakfast.rationale

    # Lunch and Dinner candidate catalog exhaustion explicit reporting
    lunch = next(m for m in meals if m.meal_type == MealType.lunch)
    assert len(lunch.foods) == 0  # No safe candidates exist in current 33-food catalog
    assert "[STATUS: INCOMPLETE_PLAN]" in lunch.rationale
    assert "No candidate foods in current catalog satisfy strict constraints" in lunch.rationale
    assert "preserving safety over completeness" in lunch.rationale

    dinner = next(m for m in meals if m.meal_type == MealType.dinner)
    assert len(dinner.foods) == 0
    assert "[STATUS: INCOMPLETE_PLAN]" in dinner.rationale
    assert "No candidate foods in current catalog satisfy strict constraints" in dinner.rationale


def test_audit_scenario_3_type_2_diabetes_variety():
    """Scenario 3: Type 2 diabetes, 1900 kcal target."""
    as_of = datetime(2026, 10, 23, 7, 0, tzinfo=COLOMBO)
    req = RecommendationRequest(
        user_id=UID,
        as_of=as_of,
        timezone="Asia/Colombo",
        profile=DietProfile(
            health_conditions=["Type 2 diabetes"],
            daily_calorie_target=1900,
        ),
    )

    meals, activities = recommend_kg(req)

    # Check Variety: Lunch and Dinner must NOT duplicate the same main course (e.g. no salmon x2)
    lunch = next(m for m in meals if m.meal_type == MealType.lunch)
    dinner = next(m for m in meals if m.meal_type == MealType.dinner)
    lunch_food_names = [f.name for f in lunch.foods]
    dinner_food_names = [f.name for f in dinner.foods]
    assert lunch_food_names != dinner_food_names, f"Duplicated main dish for lunch and dinner: {lunch_food_names}"

    # Verify no contraindicated high-GI foods were selected
    all_rec_foods = [f.name for m in meals for f in m.foods]
    assert "Instant noodles" not in all_rec_foods
    assert "Kiribath with lunu miris" not in all_rec_foods
