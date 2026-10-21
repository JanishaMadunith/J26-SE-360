"""Ground Truth Evaluation Scenarios: Based on Table 8 of the Project Proposal."""

from __future__ import annotations

from dataclasses import dataclass
from .schema import RawUserContext


@dataclass
class EvaluationTestCase:
    scenario_id: str
    description: str
    context: RawUserContext
    expected_food_ids: list[str]       # Ground-truth relevant foods
    disallowed_food_ids: list[str]     # Strictly forbidden foods (safety violation if recommended)
    expected_task_ids: list[str]       # Ground-truth relevant wellness tasks


TEST_SCENARIOS: list[EvaluationTestCase] = [
    EvaluationTestCase(
        scenario_id="SCENARIO_1",
        description="High stress + low focus + tight schedule (15 min free time, lunch)",
        context=RawUserContext(
            user_id="user_stress_tight",
            affective_state="stress",
            mental_fatigue=0.7,
            energy_level=0.4,
            focus_level=0.3,
            available_free_minutes=15,
            meal_type="lunch",
            target_meal_calories=400,
        ),
        expected_food_ids=[
            "food:red_rice_with_dhal_curry_and_gotukola_sambol",
            "food:red_rice_with_fish_curry_and_mallum",
            "food:lentil_soup_with_wholegrain_bread",
            "food:kankun_water_spinach_stir-fry_with_rice",
        ],
        disallowed_food_ids=[
            "food:instant_noodles",
        ],
        expected_task_ids=[
            "task:breathing_478",
            "task:box_breathing",
            "task:walk_outside",
        ],
    ),
    EvaluationTestCase(
        scenario_id="SCENARIO_2",
        description="Low energy + high mental fatigue + 30 min free time (afternoon snack)",
        context=RawUserContext(
            user_id="user_fatigue_afternoon",
            affective_state="neutral",
            mental_fatigue=0.85,
            energy_level=0.25,
            focus_level=0.4,
            available_free_minutes=30,
            meal_type="snack",
            target_meal_calories=250,
        ),
        expected_food_ids=[
            "food:boiled_chickpeas_kadala_tempered",
            "food:boiled_mung_beans_green_gram_with_coconut",
            "food:dark_chocolate_70%+",
        ],
        disallowed_food_ids=[
            "food:instant_noodles",
        ],
        expected_task_ids=[
            "task:rest_break",
            "task:walk_outside",
            "task:hydration_pause",
        ],
    ),
    EvaluationTestCase(
        scenario_id="SCENARIO_3",
        description="Diabetic user + elevated anxiety + lunch (Clinical constraint check)",
        context=RawUserContext(
            user_id="user_diabetic_anxious",
            affective_state="anxiety",
            mental_fatigue=0.5,
            energy_level=0.6,
            focus_level=0.5,
            available_free_minutes=30,
            meal_type="lunch",
            health_conditions=["diabetes"],
            target_meal_calories=550,
        ),
        expected_food_ids=[
            "food:red_rice_with_dhal_curry_and_gotukola_sambol",
            "food:red_rice_with_fish_curry_and_mallum",
            "food:baked_salmon_with_vegetables",
            "food:kankun_water_spinach_stir-fry_with_rice",
        ],
        disallowed_food_ids=[
            "food:instant_noodles",  # High-GI: must be blocked by KG
        ],
        expected_task_ids=[
            "task:breathing_478",
            "task:walk_outside",
            "task:box_breathing",
        ],
    ),
    EvaluationTestCase(
        scenario_id="SCENARIO_4",
        description="Strict Vegan + Fish/Egg Allergy + mental fatigue + breakfast (30 min)",
        context=RawUserContext(
            user_id="user_vegan_allergic",
            affective_state="neutral",
            mental_fatigue=0.75,
            energy_level=0.35,
            focus_level=0.4,
            available_free_minutes=30,
            meal_type="breakfast",
            is_vegan=True,
            allergies=["fish", "egg"],
            target_meal_calories=350,
        ),
        expected_food_ids=[
            "food:kola_kenda_herbal_porridge",
            "food:boiled_mung_beans_green_gram_with_coconut",
            "food:pittu_with_coconut_milk_and_dhal",
        ],
        disallowed_food_ids=[
            "food:egg_hoppers",
            "food:mackerel_tin_fish_curry_with_string_hoppers",
        ],
        expected_task_ids=[
            "task:rest_break",
            "task:hydration_pause",
            "task:walk_outside",
        ],
    ),
]
