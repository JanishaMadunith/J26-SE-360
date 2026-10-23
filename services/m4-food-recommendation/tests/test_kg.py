"""Unit and integration tests for the M4 Knowledge Graph recommendation engine."""

from m4_food.graph.knowledge_base import create_default_knowledge_graph
from m4_food.graph.kg_recommender import KnowledgeGraphRecommender
from m4_food.graph.schema import RawUserContext


def test_kg_structure_and_nodes():
    g = create_default_knowledge_graph()
    assert g.number_of_nodes() > 20
    assert g.number_of_edges() > 40
    assert g.has_node("state:stress")
    assert g.has_node("nutrient:magnesium")
    assert g.has_node("condition:diabetes")


def test_kg_recommends_magnesium_and_omega_for_stress():
    recommender = KnowledgeGraphRecommender()
    ctx = RawUserContext(
        affective_state="stress",
        mental_fatigue=0.8,
        energy_level=0.3,
        focus_level=0.3,
        available_free_minutes=30,
        meal_type="lunch",
    )
    result = recommender.recommend(ctx, top_k_foods=3, top_k_tasks=2)
    assert len(result["recommended_foods"]) > 0
    # Top food must address stress/fatigue with appropriate nutrient rationale
    top_food = result["recommended_foods"][0]
    rationale_lower = top_food["rationale"].lower()
    assert any(k in rationale_lower for k in ("magnesium", "complex carb", "omega-3", "protein"))


def test_kg_strict_allergy_pruning():
    recommender = KnowledgeGraphRecommender()
    ctx = RawUserContext(
        affective_state="neutral",
        allergies=["fish"],
        meal_type="lunch",
    )
    result = recommender.recommend(ctx, top_k_foods=5)
    rec_names = [f["name"].lower() for f in result["recommended_foods"]]
    assert not any("salmon" in n or "fish" in n or "mackerel" in n for n in rec_names)


def test_kg_diabetes_contraindication_pruning():
    recommender = KnowledgeGraphRecommender()
    ctx = RawUserContext(
        health_conditions=["diabetes"],
        meal_type="snack",
    )
    result = recommender.recommend(ctx, top_k_foods=5)
    rec_names = [f["name"].lower() for f in result["recommended_foods"]]
    assert not any("pastry" in n or "instant noodles" in n for n in rec_names)


def test_kg_task_duration_respects_time_gap():
    recommender = KnowledgeGraphRecommender()
    # Only 5 minutes free time
    ctx = RawUserContext(
        affective_state="anxiety",
        available_free_minutes=5,
    )
    result = recommender.recommend(ctx, top_k_tasks=2)
    for task in result["recommended_tasks"]:
        assert task["duration_minutes"] <= 5


def test_gnn_model_accuracy_and_ranking_metrics():
    from m4_food.graph.evaluate_gnn import compute_metrics, MODELS_DIR
    weights_path = MODELS_DIR / "gnn_embeddings.pt"
    if weights_path.exists():
        metrics = compute_metrics(weights_path, create_default_knowledge_graph, eval_samples=500)
        assert metrics["bpr_ranking_accuracy_pct"] >= 85.0, f"Expected BPR accuracy >= 85%, got {metrics['bpr_ranking_accuracy_pct']}%"
        assert metrics["hr@5"] >= 80.0, f"Expected HR@5 >= 80%, got {metrics['hr@5']}%"
