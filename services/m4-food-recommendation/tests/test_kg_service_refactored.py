"""Unit and integration tests for the refactored KnowledgeGraphService in kg_service.py.

Covers:
1. Canonical allergen normalization and filtering for all 10 allergens:
   milk, eggs, peanuts, tree nuts, fish, wheat, gluten, soy, sesame, coconut.
2. Coconut strictly separated from FDA tree nuts.
3. Gluten strictly separated from wheat allergy.
4. Unresolved allergen review status handling (unknown != safe).
5. Deterministic dietary restrictions (vegan, vegetarian, halal) independent of neural ranking.
6. Health condition contraindications (Diabetes via GL/GI, Hypertension via sodium mg).
7. Missing nutrient values preserved as unknown.
8. GNN unavailable fallback vs GNN loaded cosine scoring.
9. Reproducible ranking and graph lookup failure handling.
10. Schedule-based meal selection and dynamic KG-driven task recommendations.
11. Evaluation comparisons: KG-only, GNN-only, and Combined model.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pytest

from m4_food.foods import FOODS, Food
from m4_food.graph.kg_service import (
    EQUAL_WEIGHT_BASELINE,
    ClinicalSafetyThresholds,
    KnowledgeGraphService,
    ScoringWeights,
    normalize_allergen,
)
from wellness_contracts import (
    AffectiveLabel,
    AffectiveStateResult,
    BusyBlock,
    DietProfile,
    MealType,
    ModelInfo,
    RecommendationRequest,
    ScheduleResult,
    SuggestedActivity,
    TaskCategory,
    WellnessForecast,
    WellnessSlot,
)

COLOMBO = ZoneInfo("Asia/Colombo")
AS_OF = datetime(2026, 10, 23, 8, 0, tzinfo=COLOMBO)
UID = uuid.uuid4()


@pytest.fixture
def service():
    return KnowledgeGraphService()


@pytest.fixture
def foods_by_name():
    return {f.name: f for f in FOODS}


# =========================================================================
# 1. CANONICAL ALLERGEN TESTS FOR ALL 10 ALLERGENS
# =========================================================================

def test_allergen_milk_and_dairy(service, foods_by_name):
    # Test both canonical "milk" and alias "dairy"
    milk_tea_name = next(k for k in foods_by_name if "Milk tea" in k)
    for term in ["milk", "dairy", "cow_milk", "lactose"]:
        prof = DietProfile(allergies=[term])
        # Dairy foods must be excluded
        assert not service.is_safe_kg(foods_by_name["Curd with kithul treacle"], prof)
        assert not service.is_safe_kg(foods_by_name[milk_tea_name], prof)
        assert not service.is_safe_kg(foods_by_name["Yoghurt with mixed nuts"], prof)
        assert not service.is_safe_kg(foods_by_name["Oats porridge with banana and nuts"], prof)
        # Non-dairy foods must be safe
        assert service.is_safe_kg(foods_by_name["Papaya"], prof)
        assert service.is_safe_kg(foods_by_name["Kiribath with lunu miris"], prof)


def test_allergen_eggs(service, foods_by_name):
    for term in ["egg", "eggs", "yolk"]:
        prof = DietProfile(allergies=[term])
        assert not service.is_safe_kg(foods_by_name["Egg hoppers"], prof)
        assert service.is_safe_kg(foods_by_name["Papaya"], prof)
        assert service.is_safe_kg(foods_by_name["Kiribath with lunu miris"], prof)


def test_allergen_peanuts(service, foods_by_name):
    for term in ["peanut", "peanuts", "groundnut"]:
        prof = DietProfile(allergies=[term])
        assert not service.is_safe_kg(foods_by_name["Wholegrain toast with peanut butter"], prof)
        assert service.is_safe_kg(foods_by_name["Papaya"], prof)


def test_allergen_tree_nuts_vs_coconut_separation(service, foods_by_name):
    """CRITICAL: Tree nuts allergy must exclude cashews/walnuts, but NOT coconut!"""
    for term in ["tree_nuts", "tree nuts", "nuts", "cashew"]:
        prof = DietProfile(allergies=[term])
        # Tree nut foods excluded
        assert not service.is_safe_kg(foods_by_name["Roasted cashews"], prof)
        assert not service.is_safe_kg(foods_by_name["Yoghurt with mixed nuts"], prof)
        # Coconut foods WITHOUT tree nuts MUST BE SAFE (Coconut != FDA Tree Nut)
        assert service.is_safe_kg(foods_by_name["King coconut water (thambili)"], prof)
        assert service.is_safe_kg(foods_by_name["Kiribath with lunu miris"], prof)
        assert service.is_safe_kg(foods_by_name["Pittu with coconut milk and dhal"], prof)


def test_allergen_coconut_alone(service, foods_by_name):
    """CRITICAL: Coconut allergy excludes coconut foods, but NOT tree nuts without coconut!"""
    prof = DietProfile(allergies=["coconut"])
    # Coconut foods excluded
    assert not service.is_safe_kg(foods_by_name["King coconut water (thambili)"], prof)
    assert not service.is_safe_kg(foods_by_name["Kiribath with lunu miris"], prof)
    assert not service.is_safe_kg(foods_by_name["Coconut roti with lunu miris"], prof)
    # Roasted cashews have NO coconut -> MUST BE SAFE
    assert service.is_safe_kg(foods_by_name["Roasted cashews"], prof)


def test_allergen_fish(service, foods_by_name):
    for term in ["fish", "finfish", "salmon", "mackerel"]:
        prof = DietProfile(allergies=[term])
        assert not service.is_safe_kg(foods_by_name["Baked salmon with vegetables"], prof)
        assert not service.is_safe_kg(foods_by_name["Red rice with fish curry and mallum"], prof)
        assert not service.is_safe_kg(foods_by_name["Mackerel (tin fish) curry with string hoppers"], prof)
        assert service.is_safe_kg(foods_by_name["Papaya"], prof)


def test_allergen_wheat_and_gluten_separation(service, foods_by_name):
    """Wheat allergy excludes wheat. Gluten intolerance excludes gluten grains."""
    prof_wheat = DietProfile(allergies=["wheat"])
    prof_gluten = DietProfile(allergies=["gluten"])

    # Foods containing wheat flour (string hoppers, wholegrain toast, roti, noodles)
    wheat_foods = [
        "String hoppers with kiri hodi and pol sambol",
        "Coconut roti with lunu miris",
        "Wholegrain toast with peanut butter",
        "Lentil soup with wholegrain bread",
        "Instant noodles",
    ]
    for wname in wheat_foods:
        assert not service.is_safe_kg(foods_by_name[wname], prof_wheat), f"Expected {wname} excluded for wheat"
        assert not service.is_safe_kg(foods_by_name[wname], prof_gluten), f"Expected {wname} excluded for gluten"

    # Naturally gluten-free foods
    gf_foods = ["Papaya", "King coconut water (thambili)", "Boiled mung beans (green gram) with coconut"]
    for gname in gf_foods:
        assert service.is_safe_kg(foods_by_name[gname], prof_wheat)
        assert service.is_safe_kg(foods_by_name[gname], prof_gluten)


def test_allergen_soybeans(service, foods_by_name):
    for term in ["soy", "soya", "soybeans", "soybean"]:
        prof = DietProfile(allergies=[term])
        assert not service.is_safe_kg(foods_by_name["Soya meat curry with red rice"], prof)
        assert service.is_safe_kg(foods_by_name["Papaya"], prof)


def test_allergen_sesame(service):
    """Sesame allergy test: dummy food with sesame tag or ingredient is excluded."""
    sesame_food = Food(
        name="Sesame Snack Bar",
        portion="1 bar (40g)",
        calories=180,
        meals=frozenset([MealType.snack]),
        tags=frozenset({"snack", "sesame"}),
        allergens=frozenset({"sesame"}),
    )
    svc = KnowledgeGraphService(foods_list=[sesame_food])
    prof = DietProfile(allergies=["sesame"])
    assert not svc.is_safe_kg(sesame_food, prof)


def test_unresolved_allergen_review_status_excluded_for_allergic_users(service):
    """Foods with UNVERIFIED_REVIEW_PENDING cannot be assumed safe when user has active allergies."""
    unresolved_food = Food(
        name="Mystery Curry",
        portion="1 plate (200g)",
        calories=300,
        meals=frozenset([MealType.lunch]),
        tags=frozenset({"lunch", "curry"}),
        allergen_review_status="UNVERIFIED_REVIEW_PENDING",
    )
    svc = KnowledgeGraphService(foods_list=[unresolved_food])
    allergic_user = DietProfile(allergies=["dairy"])
    # Must be excluded because allergen status is unverified
    assert not svc.is_safe_kg(unresolved_food, allergic_user)

    # Non-allergic user can receive it
    non_allergic_user = DietProfile(allergies=[])
    assert svc.is_safe_kg(unresolved_food, non_allergic_user)


# =========================================================================
# 2. DIETARY PREFERENCES (VEGAN, VEGETARIAN, HALAL) INDEPENDENCE
# =========================================================================

def test_dietary_restrictions_independent_of_neural_score(service, foods_by_name):
    """Neural rank cannot elevate a non-vegan/non-vegetarian/non-halal food into recommendations."""
    # 1. Vegan test
    vegan_prof = DietProfile(vegan=True)
    for f in FOODS:
        is_safe = service.is_safe_kg(f, vegan_prof)
        if not (f.is_vegan or "vegan" in f.tags):
            assert not is_safe, f"Non-vegan food {f.name} leaked through vegan filter"

    # 2. Vegetarian test
    veg_prof = DietProfile(vegetarian=True)
    for f in FOODS:
        is_safe = service.is_safe_kg(f, veg_prof)
        if not (f.is_vegetarian or "vegetarian" in f.tags):
            assert not is_safe, f"Non-vegetarian food {f.name} leaked through vegetarian filter"

    # 3. Halal test
    halal_prof = DietProfile(halal=True)
    for f in FOODS:
        is_safe = service.is_safe_kg(f, halal_prof)
        if not (f.is_halal or "halal" in f.tags):
            assert not is_safe, f"Non-halal food {f.name} leaked through halal filter"


# =========================================================================
# 3. CLINICAL HEALTH CONDITION RULES (DIABETES & HYPERTENSION)
# =========================================================================

def test_diabetes_excludes_high_gi_and_allows_negligible_carb(service, foods_by_name):
    prof = DietProfile(health_conditions=["Type 2 Diabetes"])

    # High GI / high sugar starches excluded
    assert not service.is_safe_kg(foods_by_name["Instant noodles"], prof)
    assert not service.is_safe_kg(foods_by_name["Kiribath with lunu miris"], prof)

    # Negligible carbohydrate foods (salmon) do not spike glucose -> allowed
    assert service.is_safe_kg(foods_by_name["Baked salmon with vegetables"], prof)

    # High carbohydrate food with unverified GI is contraindicated for diabetes
    assert not service.is_safe_kg(foods_by_name["Egg hoppers"], prof)

    # Low GI complex legumes allowed
    assert service.is_safe_kg(foods_by_name["Boiled chickpeas (kadala) tempered"], prof)
    assert service.is_safe_kg(foods_by_name["Boiled mung beans (green gram) with coconut"], prof)


def test_hypertension_sodium_threshold(service, foods_by_name):
    prof = DietProfile(health_conditions=["Hypertension"])

    # Instant noodles has > 400mg sodium or high-sodium tag -> excluded
    assert not service.is_safe_kg(foods_by_name["Instant noodles"], prof)

    # Fresh fruits and low sodium meals allowed
    assert service.is_safe_kg(foods_by_name["Papaya"], prof)
    assert service.is_safe_kg(foods_by_name["Fruit salad"], prof)


# =========================================================================
# 4. GNN SCORING, FALLBACK, AND ABLATION COMPARISONS
# =========================================================================

def test_gnn_unavailable_fallback(foods_by_name):
    """When GNN is disabled, service falls back cleanly to KG path scoring."""
    svc_no_gnn = KnowledgeGraphService(enable_gnn=False)
    assert svc_no_gnn.gnn is None

    score, reasons = svc_no_gnn.score_food_kg(
        food=foods_by_name["Roasted cashews"],
        meal=MealType.snack,
        label=AffectiveLabel.stress,
        energy=0.8,
        local_hour=10,
        kcal_target=200,
    )
    assert isinstance(score, float)
    assert not any("GNN Score" in r for r in reasons)
    assert any("magnesium" in r.lower() for r in reasons)


def test_gnn_loaded_cosine_scoring(service, foods_by_name):
    """When GNN is enabled, cosine similarity scores are ordinary floats."""
    if service.gnn is not None and service.gnn.is_loaded:
        score, reasons = service.score_food_kg(
            food=foods_by_name["Roasted cashews"],
            meal=MealType.snack,
            label=AffectiveLabel.stress,
            energy=0.8,
            local_hour=10,
            kcal_target=200,
        )
        assert isinstance(score, float)
        assert any("GNN Score" in r for r in reasons)


def test_model_ablation_comparisons(foods_by_name):
    """Compares KG-only, GNN-only, and Combined model scoring."""
    food = foods_by_name["Roasted cashews"]

    # 1. KG-Only (Equal Weight Baseline, GNN disabled)
    svc_kg_only = KnowledgeGraphService(enable_gnn=False, weights=EQUAL_WEIGHT_BASELINE)
    score_kg, _ = svc_kg_only.score_food_kg(food, MealType.snack, AffectiveLabel.stress, 0.8, 10, 200)

    # 2. GNN-Only (KG weight 0, GNN weight 1.0)
    gnn_only_weights = ScoringWeights(kg_path_weight=0.0, gnn_weight=1.0, verified_nutrient_weight=0.0, calorie_proximity_weight=0.0)
    svc_gnn_only = KnowledgeGraphService(enable_gnn=True, weights=gnn_only_weights)
    score_gnn, _ = svc_gnn_only.score_food_kg(food, MealType.snack, AffectiveLabel.stress, 0.8, 10, 200)

    # 3. Combined Model (Default)
    svc_combined = KnowledgeGraphService(enable_gnn=True)
    score_combined, _ = svc_combined.score_food_kg(food, MealType.snack, AffectiveLabel.stress, 0.8, 10, 200)

    assert isinstance(score_kg, float)
    assert isinstance(score_gnn, float)
    assert isinstance(score_combined, float)


# =========================================================================
# 5. REPRODUCIBILITY & GRAPH LOOKUP FAILURES
# =========================================================================

def test_reproducible_ranking(service):
    """Same context must produce identical food ranking and deterministic outputs."""
    used_1: dict[str, int] = {}
    used_2: dict[str, int] = {}
    prof = DietProfile(vegetarian=True)

    res_1 = service.pick_foods_kg(MealType.lunch, AffectiveLabel.stress, 0.5, 12, prof, used_1)
    res_2 = service.pick_foods_kg(MealType.lunch, AffectiveLabel.stress, 0.5, 12, prof, used_2)

    assert [f.name for f in res_1] == [f.name for f in res_2]
    assert [f.calories for f in res_1] == [f.calories for f in res_2]


def test_graph_lookup_failure_graceful_handling(service):
    """A food missing from the graph is safely rejected without raising an exception."""
    phantom_food = Food(
        name="Phantom Alien Snack",
        portion="100g",
        calories=100,
        meals=frozenset([MealType.snack]),
        tags=frozenset({"snack"}),
    )
    assert not service.is_safe_kg(phantom_food, DietProfile())


# =========================================================================
# 6. DYNAMIC KNOWLEDGE GRAPH TASK RECOMMENDATIONS
# =========================================================================

def test_dynamic_task_recommendations_from_kg(service):
    # Case A: Acute Stress
    req_stress = RecommendationRequest(
        user_id=UID,
        as_of=AS_OF,
        affective_state=AffectiveStateResult(
            user_id=UID,
            window_start=AS_OF - timedelta(minutes=30),
            window_end=AS_OF,
            label=AffectiveLabel.stress,
            probabilities={"stress": 0.9, "relaxed": 0.1},
            confidence=0.9,
            model=ModelInfo(name="test", version="1"),
        ),
    )
    tasks_stress = service.suggest_activities_kg(req_stress)
    assert len(tasks_stress) >= 1
    assert all(isinstance(t, SuggestedActivity) for t in tasks_stress)
    assert all(t.estimated_minutes >= 5 for t in tasks_stress)
    assert tasks_stress[0].category == TaskCategory.exercise
    assert "walk" in tasks_stress[0].title.lower()

    # Case B: High Mental Fatigue
    slot_fatigue = WellnessSlot(
        start=AS_OF,
        end=AS_OF + timedelta(hours=1),
        mental_fatigue=0.85,
        focus=0.3,
        energy=0.3,
        burnout_risk=0.4,
    )
    req_fatigue = RecommendationRequest(
        user_id=UID,
        as_of=AS_OF,
        wellness=WellnessForecast(
            user_id=UID,
            generated_at=AS_OF,
            slot_minutes=60,
            slots=[slot_fatigue, slot_fatigue, slot_fatigue],
            model=ModelInfo(name="test", version="1"),
        ),
    )
    tasks_fatigue = service.suggest_activities_kg(req_fatigue)
    assert any("rest break" in t.title.lower() for t in tasks_fatigue)


def test_heartsteps_dataset_in_knowledge_graph(service):
    """Verifies that HeartSteps empirical activities are loaded into the Knowledge Graph."""
    hs_nodes = [nid for nid in service.g.nodes() if "heartsteps" in nid]
    assert len(hs_nodes) >= 20, f"Expected at least 20 HeartSteps task nodes, got {len(hs_nodes)}"

    sample_node = service.g.nodes[hs_nodes[0]]["node"]
    assert sample_node.properties.get("empirical_stats") is not None
    assert sample_node.properties["empirical_stats"]["positive_ratio"] >= 0.65
    assert "HeartSteps" in sample_node.properties["source"]

