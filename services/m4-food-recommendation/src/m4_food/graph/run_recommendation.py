

"""Interactive CLI Recommendation Runner for M4.

Allows entering:
- Affective State (stress, anxiety, sad, relaxed, neutral)
- Wellness (mental fatigue, energy, focus)
- Schedule (free time duration in minutes, meal slot)
- Profile (vegan, vegetarian, halal, allergies, health conditions, target calories)

Outputs personalized foods and wellness tasks reasoned via Knowledge Graph + GNN.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure package on path
src_dir = str(Path(__file__).resolve().parents[2])
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from m4_food.graph.kg_recommender import KnowledgeGraphRecommender
from m4_food.graph.schema import RawUserContext


def prompt_interactive() -> RawUserContext:
    print("\n" + "=" * 70)
    print("      CONTEXT-AWARE FOOD & TASK RECOMMENDATION SYSTEM (M4)")
    print("=" * 70)
    print("Please enter your current context below (press Enter to accept default):\n")

    # 1. Affective State
    print("[1] AFFECTIVE STATE")
    aff = input("  Current mood / emotion [stress / anxiety / sad / relaxed / neutral] (default: stress): ").strip() or "stress"

    # 2. Wellness Metrics
    print("\n[2] COGNITIVE WELLNESS (0.0 to 1.0 scale)")
    fatigue_raw = input("  Mental fatigue [0.0 = fresh, 1.0 = exhausted] (default: 0.8): ").strip()
    energy_raw = input("  Energy level   [0.0 = drained, 1.0 = energized] (default: 0.3): ").strip()
    focus_raw = input("  Focus level    [0.0 = distracted, 1.0 = sharp] (default: 0.4): ").strip()

    fatigue = float(fatigue_raw) if fatigue_raw else 0.8
    energy = float(energy_raw) if energy_raw else 0.3
    focus = float(focus_raw) if focus_raw else 0.4

    # 3. Schedule
    print("\n[3] SCHEDULE & TIME")
    meal = input("  Meal type [breakfast / lunch / snack / dinner] (default: lunch): ").strip() or "lunch"
    time_raw = input("  Available free time in minutes (default: 20): ").strip()
    available_min = int(time_raw) if time_raw else 20

    # 4. Profile & Health
    print("\n[4] DIETARY PROFILE & HEALTH")
    vegan = input("  Are you vegan? [y/N]: ").strip().lower() in ("y", "yes", "true")
    vegetarian = input("  Are you vegetarian? [y/N]: ").strip().lower() in ("y", "yes", "true")
    halal = input("  Do you require halal? [y/N]: ").strip().lower() in ("y", "yes", "true")
    allergies_raw = input("  Allergies (comma-separated, e.g. fish, nuts, dairy) [None]: ").strip()
    allergies = [a.strip() for a in allergies_raw.split(",") if a.strip()]

    conditions_raw = input("  Health conditions (e.g. diabetes, hypertension) [None]: ").strip()
    conditions = [c.strip() for c in conditions_raw.split(",") if c.strip()]

    cal_raw = input("  Target meal calories (default: 500 kcal): ").strip()
    target_cal = int(cal_raw) if cal_raw else 500

    return RawUserContext(
        user_id="user_interactive",
        affective_state=aff,
        mental_fatigue=fatigue,
        energy_level=energy,
        focus_level=focus,
        available_free_minutes=available_min,
        meal_type=meal,
        is_vegan=vegan,
        is_vegetarian=vegetarian,
        is_halal=halal,
        allergies=allergies,
        health_conditions=conditions,
        target_meal_calories=target_cal,
    )


def display_recommendations(ctx: RawUserContext, rec_engine: KnowledgeGraphRecommender):
    print("\n" + "=" * 70)
    print("                    ANALYZING USER CONTEXT")
    print("=" * 70)
    print(f"  * Affective State    : {ctx.affective_state.capitalize()}")
    print(f"  * Cognitive Forecast : Fatigue={ctx.mental_fatigue:.2f}, Energy={ctx.energy_level:.2f}, Focus={ctx.focus_level:.2f}")
    print(f"  * Schedule Slot      : {ctx.available_free_minutes} min free | Meal: {ctx.meal_type.capitalize()} (~{ctx.target_meal_calories} kcal)")
    diet_str = []
    if ctx.is_vegan: diet_str.append("Vegan")
    if ctx.is_vegetarian: diet_str.append("Vegetarian")
    if ctx.is_halal: diet_str.append("Halal")
    print(f"  * Diet Preference    : {', '.join(diet_str) or 'Standard'}")
    print(f"  * Allergies Avoided  : {', '.join(ctx.allergies) or 'None'}")
    print(f"  * Medical Conditions : {', '.join(ctx.health_conditions) or 'None'}")
    print("=" * 70)

    res = rec_engine.recommend(ctx, top_k_foods=3, top_k_tasks=2)

    print("\n>>> PERSONALIZED FOOD RECOMMENDATIONS (REASONED VIA KG + GNN):")
    if not res["recommended_foods"]:
        print("  [!] No foods matched your safety constraints and meal criteria.")
    for idx, f in enumerate(res["recommended_foods"], 1):
        print(f"\n  [{idx}] {f['name']}")
        print(f"      Calories : {f['calories']} kcal | Prep Time: {f['prep_minutes']} min | Model Score: {f['score']}")
        print(f"      Reason   : {f['rationale']}")

    print("\n>>> SUGGESTED WELLNESS & PRODUCTIVITY TASKS:")
    if not res["recommended_tasks"]:
        print("  [!] No tasks matched the available time duration.")
    for idx, t in enumerate(res["recommended_tasks"], 1):
        print(f"\n  [{idx}] {t['title']} ({t['duration_minutes']} min, Category: {t['category']})")
        print(f"      Reason   : {t['rationale']}")

    print("\n" + "=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Get Food & Task recommendations for custom context")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive step-by-step prompt")
    parser.add_argument("--state", type=str, default=None, help="Affective state (stress, anxiety, sad, neutral)")
    parser.add_argument("--fatigue", type=float, default=0.8, help="Mental fatigue (0.0 to 1.0)")
    parser.add_argument("--energy", type=float, default=0.3, help="Energy level (0.0 to 1.0)")
    parser.add_argument("--focus", type=float, default=0.4, help="Focus level (0.0 to 1.0)")
    parser.add_argument("--time", type=int, default=25, help="Available free minutes from schedule")
    parser.add_argument("--meal", type=str, default="lunch", help="Meal type (breakfast, lunch, snack, dinner)")
    parser.add_argument("--allergies", type=str, default="", help="Comma-separated allergies")
    parser.add_argument("--conditions", type=str, default="", help="Comma-separated conditions (e.g. diabetes)")
    parser.add_argument("--vegan", action="store_true", help="User is vegan")
    parser.add_argument("--vegetarian", action="store_true", help="User is vegetarian")
    parser.add_argument("--halal", action="store_true", help="User requires halal")
    parser.add_argument("--calories", type=int, default=500, help="Target calories")
    parser.add_argument("--expanded", action="store_true", help="Use expanded multi-source Knowledge Graph (132 nodes, Food.com + USDA)")

    args = parser.parse_args()

    if args.expanded:
        from m4_food.graph.knowledge_base_expanded import create_expanded_knowledge_graph
        from m4_food.graph.gnn_ranker import GNNRecommenderRanker
        from pathlib import Path
        expanded_weights = Path(__file__).resolve().parents[3] / "models" / "gnn_embeddings_expanded.pt"
        ranker = GNNRecommenderRanker(weights_path=expanded_weights) if expanded_weights.exists() else None
        engine = KnowledgeGraphRecommender(graph=create_expanded_knowledge_graph())
        engine.gnn = ranker
        print(">>> [ENGINE MODE] Running on EXPANDED Multi-Source Knowledge Graph (132 Nodes, Food.com + USDA)")
    else:
        engine = KnowledgeGraphRecommender()
        print(">>> [ENGINE MODE] Running on BASELINE Validated Knowledge Graph (72 Nodes, Sri Lanka + USDA)")

    if args.interactive or (args.state is None and len(sys.argv) == 1):
        ctx = prompt_interactive()
    else:
        ctx = RawUserContext(
            user_id="user_cli",
            affective_state=args.state or "stress",
            mental_fatigue=args.fatigue,
            energy_level=args.energy,
            focus_level=args.focus,
            available_free_minutes=args.time,
            meal_type=args.meal,
            is_vegan=args.vegan,
            is_vegetarian=args.vegetarian,
            is_halal=args.halal,
            allergies=[a.strip() for a in args.allergies.split(",") if a.strip()],
            health_conditions=[c.strip() for c in args.conditions.split(",") if c.strip()],
            target_meal_calories=args.calories,
        )

    display_recommendations(ctx, engine)


if __name__ == "__main__":
    main()
