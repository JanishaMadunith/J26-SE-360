"""Interactive Tester & Benchmark Evaluator for M4 Knowledge Graph Recommendations.

Run from repository root:
    python -m services.m4-food-recommendation.src.m4_food.graph.interactive_tester --eval
or
    python -m services.m4-food-recommendation.src.m4_food.graph.interactive_tester --raw
"""

from __future__ import annotations

import argparse
import time
from typing import Any

from .kg_recommender import KnowledgeGraphRecommender
from .scenarios import TEST_SCENARIOS, EvaluationTestCase
from .schema import RawUserContext


def evaluate_benchmark(recommender: KnowledgeGraphRecommender, k_food: int = 2, k_task: int = 2) -> dict[str, Any]:
    """Runs all test scenarios and computes Precision@K, Recall@K, HitRate@K, and Safety Violation Rate."""
    print("=" * 80)
    print(f"RUNNING BENCHMARK EVALUATION (K_food={k_food}, K_task={k_task})")
    print("=" * 80)

    food_precisions = []
    food_recalls = []
    food_hits = []
    task_precisions = []
    task_recalls = []
    task_hits = []
    safety_violations = 0
    total_recs = 0
    latencies = []

    for test_case in TEST_SCENARIOS:
        print(f"\n[Test Case] {test_case.scenario_id}: {test_case.description}")
        start_t = time.perf_counter()
        result = recommender.recommend(test_case.context, top_k_foods=k_food, top_k_tasks=k_task)
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        latencies.append(elapsed_ms)

        rec_food_ids = [f["food_id"] for f in result["recommended_foods"]]
        rec_task_ids = [t["task_id"] for t in result["recommended_tasks"]]

        # 1. Food Evaluation
        exp_foods = set(test_case.expected_food_ids)
        rel_foods_found = [fid for fid in rec_food_ids if fid in exp_foods]
        f_prec = len(rel_foods_found) / max(len(rec_food_ids), 1)
        f_rec = len(rel_foods_found) / max(len(exp_foods), 1)
        f_hit = 1.0 if len(rel_foods_found) > 0 else 0.0

        food_precisions.append(f_prec)
        food_recalls.append(f_rec)
        food_hits.append(f_hit)

        # 2. Safety / Constraint Check
        for fid in rec_food_ids:
            total_recs += 1
            if fid in test_case.disallowed_food_ids:
                print(f"  ❌ CRITICAL SAFETY VIOLATION: Recommended disallowed item {fid}")
                safety_violations += 1

        # 3. Task Evaluation
        exp_tasks = set(test_case.expected_task_ids)
        rel_tasks_found = [tid for tid in rec_task_ids if tid in exp_tasks]
        t_prec = len(rel_tasks_found) / max(len(rec_task_ids), 1)
        t_rec = len(rel_tasks_found) / max(len(exp_tasks), 1)
        t_hit = 1.0 if len(rel_tasks_found) > 0 else 0.0

        task_precisions.append(t_prec)
        task_recalls.append(t_rec)
        task_hits.append(t_hit)

        print(f"  Active States Detected: {result['active_context_states']}")
        print(f"  Food Recs : {[f['name'] for f in result['recommended_foods']]}")
        print(f"  Task Recs : {[t['title'] for t in result['recommended_tasks']]}")
        print(f"  Food P@{k_food}: {f_prec:.2f} | Recall: {f_rec:.2f} | Latency: {elapsed_ms:.2f} ms")

    avg_f_prec = sum(food_precisions) / len(food_precisions)
    avg_f_rec = sum(food_recalls) / len(food_recalls)
    avg_f_hit = sum(food_hits) / len(food_hits)

    avg_t_prec = sum(task_precisions) / len(task_precisions)
    avg_t_rec = sum(task_recalls) / len(task_recalls)
    avg_t_hit = sum(task_hits) / len(task_hits)

    violation_rate = (safety_violations / max(total_recs, 1)) * 100.0
    avg_latency = sum(latencies) / len(latencies)

    print("\n" + "=" * 80)
    print("BENCHMARK EVALUATION SUMMARY")
    print("=" * 80)
    print(f"  Food Precision@{k_food}          : {avg_f_prec * 100:.1f}%")
    print(f"  Food Recall@{k_food}             : {avg_f_rec * 100:.1f}%")
    print(f"  Food HitRate@{k_food}            : {avg_f_hit * 100:.1f}%")
    print(f"  Task Precision@{k_task}          : {avg_t_prec * 100:.1f}%")
    print(f"  Task Recall@{k_task}             : {avg_t_rec * 100:.1f}%")
    print(f"  Task HitRate@{k_task}            : {avg_t_hit * 100:.1f}%")
    print(f"  Constraint Safety Violation Rate : {violation_rate:.1f}% (Required: 0.0%)")
    print(f"  Average KG Latency              : {avg_latency:.2f} ms (Target: < 3000 ms)")
    print("=" * 80)

    return {
        "avg_food_precision": avg_f_prec,
        "avg_food_recall": avg_f_rec,
        "avg_food_hit_rate": avg_f_hit,
        "avg_task_precision": avg_t_prec,
        "violation_rate": violation_rate,
        "avg_latency_ms": avg_latency,
    }


def run_custom_raw_input(recommender: KnowledgeGraphRecommender, args: argparse.Namespace) -> None:
    """Runs a single query with custom raw parameters and displays full graph explanations."""
    allergies_list = [a.strip() for a in args.allergies.split(",")] if args.allergies else []
    conditions_list = [c.strip() for c in args.conditions.split(",")] if args.conditions else []

    ctx = RawUserContext(
        user_id="cli_tester",
        affective_state=args.state,
        mental_fatigue=args.fatigue,
        energy_level=args.energy,
        focus_level=args.focus,
        available_free_minutes=args.time,
        meal_type=args.meal,
        is_vegan=args.vegan,
        is_vegetarian=args.vegetarian,
        is_halal=args.halal,
        allergies=allergies_list,
        health_conditions=conditions_list,
        target_meal_calories=args.calories,
    )

    print("\n" + "=" * 80)
    print("RAW USER CONTEXT INPUT:")
    print("=" * 80)
    print(f"  Affective State   : {ctx.affective_state}")
    print(f"  Cognitive Metrics : Fatigue={ctx.mental_fatigue:.2f}, Energy={ctx.energy_level:.2f}, Focus={ctx.focus_level:.2f}")
    print(f"  Schedule Gap      : {ctx.available_free_minutes} minutes available")
    print(f"  Meal Slot         : {ctx.meal_type.capitalize()} (Target Cal: {ctx.target_meal_calories} kcal)")
    print(f"  Diet Restrictions : Vegan={ctx.is_vegan}, Veg={ctx.is_vegetarian}, Halal={ctx.is_halal}")
    print(f"  Allergies         : {ctx.allergies or 'None'}")
    print(f"  Health Conditions : {ctx.health_conditions or 'None'}")
    print("=" * 80)

    start_t = time.perf_counter()
    res = recommender.recommend(ctx, top_k_foods=args.top_k, top_k_tasks=args.top_k)
    elapsed_ms = (time.perf_counter() - start_t) * 1000.0

    print("\n[RECOMMENDED FOODS (REASONED VIA KNOWLEDGE GRAPH)]")
    if not res["recommended_foods"]:
        print("  [!] No foods matched the safety constraints and meal criteria.")
    for idx, f in enumerate(res["recommended_foods"], 1):
        print(f"  {idx}. {f['name']}")
        print(f"     * Calories : {f['calories']} kcal | Prep Time: {f['prep_minutes']} min | Score: {f['score']}")
        print(f"     * KG Rationale: {f['rationale']}\n")

    print("[SUGGESTED WELLNESS & PRODUCTIVITY TASKS]")
    if not res["recommended_tasks"]:
        print("  [!] No tasks matched the available free time duration.")
    for idx, t in enumerate(res["recommended_tasks"], 1):
        print(f"  {idx}. {t['title']} ({t['duration_minutes']} mins, Category: {t['category']})")
        print(f"     * KG Rationale: {t['rationale']}\n")

    print(f">> In-Memory KG Inference Latency: {elapsed_ms:.2f} ms")


def main():
    parser = argparse.ArgumentParser(description="Test and Evaluate M4 Knowledge Graph Recommender")
    parser.add_argument("--eval", action="store_true", help="Run benchmark evaluation over test scenarios")
    parser.add_argument("--state", type=str, default="stress", help="Affective state (e.g. stress, anxiety, neutral)")
    parser.add_argument("--fatigue", type=float, default=0.75, help="Mental fatigue score (0.0 - 1.0)")
    parser.add_argument("--energy", type=float, default=0.35, help="Energy level score (0.0 - 1.0)")
    parser.add_argument("--focus", type=float, default=0.4, help="Focus level score (0.0 - 1.0)")
    parser.add_argument("--time", type=int, default=20, help="Available free minutes from schedule")
    parser.add_argument("--meal", type=str, default="lunch", help="Meal type (breakfast, lunch, snack, dinner)")
    parser.add_argument("--calories", type=int, default=500, help="Target calories for this meal")
    parser.add_argument("--allergies", type=str, default="", help="Comma-separated allergies (e.g. fish,nuts)")
    parser.add_argument("--conditions", type=str, default="", help="Comma-separated health conditions (e.g. diabetes)")
    parser.add_argument("--vegan", action="store_true", help="User is vegan")
    parser.add_argument("--vegetarian", action="store_true", help="User is vegetarian")
    parser.add_argument("--halal", action="store_true", help="User requires halal")
    parser.add_argument("--top-k", type=int, default=2, help="Number of items to recommend")

    args = parser.parse_args()
    recommender = KnowledgeGraphRecommender()

    if args.eval:
        evaluate_benchmark(recommender)
    else:
        run_custom_raw_input(recommender, args)


if __name__ == "__main__":
    main()
