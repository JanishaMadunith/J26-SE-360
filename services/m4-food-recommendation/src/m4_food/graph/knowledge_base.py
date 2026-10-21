"""Knowledge Base: Graph Builder, Triple Definitions, and Domain Knowledge Graph."""

import json
from pathlib import Path

import networkx as nx
try:
    from m4_food.foods import FOODS, Food
except ImportError:
    from ..foods import FOODS, Food
from .schema import KGNode, NodeType, RelationType

KNOWN_DIETS = {"vegan", "vegetarian", "halal"}
KNOWN_ALLERGENS = {"dairy", "egg", "fish", "gluten", "nuts", "soy", "shellfish", "peanut", "coconut"}
KNOWN_NUTRIENTS = {
    "high-protein", "complex-carb", "omega-3", "magnesium", "low-GI",
    "high-GI", "high-sodium", "fried", "high-sugar", "caffeine", "antioxidants", "tryptophan", "potassium"
}

USDA_DATA_PATH = Path(__file__).parent / "data" / "usda_micronutrients.json"


def load_usda_nutrients_map() -> dict[str, dict]:
    """Loads parsed USDA FoodData Central micronutrient profiles."""
    if not USDA_DATA_PATH.exists():
        return {}
    try:
        with open(USDA_DATA_PATH, encoding="utf-8") as f:
            data = json.load(f)
        keyword_profiles = {}
        for entry in data.values():
            kw = entry.get("keyword", "").lower()
            if kw and kw not in keyword_profiles:
                keyword_profiles[kw] = entry.get("nutrients", {})
        return keyword_profiles
    except Exception:
        return {}


def create_default_knowledge_graph(foods_list: list[Food] | None = None) -> nx.DiGraph:
    """Builds and returns the domain Knowledge Graph (in-memory DiGraph)."""
    g = nx.DiGraph()

    # =========================================================================
    # 1. MENTAL & COGNITIVE STATES (NODES)
    # =========================================================================
    states = [
        ("state:stress", "Stress / Tension", NodeType.AFFECTIVE_STATE),
        ("state:anxiety", "Anxiety", NodeType.AFFECTIVE_STATE),
        ("state:sad", "Low Mood / Sadness", NodeType.AFFECTIVE_STATE),
        ("state:fatigue", "Mental Fatigue / Exhaustion", NodeType.COGNITIVE_STATE),
        ("state:low_energy", "Low Energy / Lethargy", NodeType.COGNITIVE_STATE),
        ("state:low_focus", "Poor Focus / Brain Fog", NodeType.COGNITIVE_STATE),
    ]
    for nid, name, ntype in states:
        g.add_node(nid, node=KGNode(id=nid, name=name, node_type=ntype))

    # =========================================================================
    # 2. NUTRIENTS & BIO-COMPOUNDS (NODES)
    # =========================================================================
    nutrients = [
        ("nutrient:magnesium", "Magnesium", NodeType.NUTRIENT, "regulates cortisol and nervous system excitability"),
        ("nutrient:omega-3", "Omega-3 Fatty Acids", NodeType.NUTRIENT, "supports cellular brain membrane and regulates mood"),
        ("nutrient:complex-carb", "Complex Carbohydrates", NodeType.NUTRIENT, "provides slow-burning glucose for steady cognitive energy"),
        ("nutrient:high-protein", "Quality Protein & Amino Acids", NodeType.NUTRIENT, "supports neurotransmitter synthesis and sustained alertness"),
        ("nutrient:antioxidants", "Polyphenols & Antioxidants", NodeType.NUTRIENT, "reduces neuroinflammation and combats mental fatigue"),
        ("nutrient:tryptophan", "L-Tryptophan", NodeType.NUTRIENT, "serotonin precursor that promotes calm emotional stability"),
        ("nutrient:potassium", "Potassium", NodeType.NUTRIENT, "helps balance electrolytes and blood pressure during stress"),
        ("nutrient:low-GI", "Low Glycemic Index", NodeType.NUTRIENT, "prevents insulin spikes and sudden mental crashes"),
        ("nutrient:high-GI", "High Glycemic Index", NodeType.NUTRIENT, "causes rapid glucose spikes and subsequent energy crashes"),
        ("nutrient:high-sodium", "High Sodium", NodeType.NUTRIENT, "elevates vascular tension and blood pressure"),
        ("nutrient:high-sugar", "High Sugar", NodeType.NUTRIENT, "triggers rapid reactive hypoglycemia and energy crashes"),
        ("nutrient:fried", "Deep Fried / High Saturated Fat", NodeType.NUTRIENT, "induces postprandial somnolence and digestive lethargy"),
        ("nutrient:caffeine", "Caffeine", NodeType.NUTRIENT, "central nervous system stimulant"),
    ]
    for nid, name, ntype, desc in nutrients:
        g.add_node(nid, node=KGNode(id=nid, name=name, node_type=ntype, properties={"description": desc}))

    # =========================================================================
    # 3. HEALTH CONDITIONS & ALLERGENS (NODES)
    # =========================================================================
    conditions = [
        ("condition:diabetes", "Diabetes", NodeType.HEALTH_CONDITION),
        ("condition:hypertension", "Hypertension", NodeType.HEALTH_CONDITION),
        ("allergen:gluten", "Gluten", NodeType.ALLERGEN),
        ("allergen:dairy", "Dairy", NodeType.ALLERGEN),
        ("allergen:fish", "Fish", NodeType.ALLERGEN),
        ("allergen:nuts", "Tree Nuts / Peanuts", NodeType.ALLERGEN),
        ("allergen:peanut", "Peanut", NodeType.ALLERGEN),
        ("allergen:coconut", "Coconut", NodeType.ALLERGEN),
        ("allergen:egg", "Egg", NodeType.ALLERGEN),
        ("allergen:soy", "Soy", NodeType.ALLERGEN),
        ("allergen:shellfish", "Shellfish", NodeType.ALLERGEN),
    ]
    for cid, name, ntype in conditions:
        g.add_node(cid, node=KGNode(id=cid, name=name, node_type=ntype))

    # =========================================================================
    # 4. DIETARY PREFERENCES (NODES)
    # =========================================================================
    diets = [
        ("diet:vegan", "Vegan", NodeType.DIETARY_PREFERENCE),
        ("diet:vegetarian", "Vegetarian", NodeType.DIETARY_PREFERENCE),
        ("diet:halal", "Halal", NodeType.DIETARY_PREFERENCE),
    ]
    for did, name, ntype in diets:
        g.add_node(did, node=KGNode(id=did, name=name, node_type=ntype))

    # =========================================================================
    # 5. NUTRIENT -> MENTAL/COGNITIVE STATE LINKS (TRIPLES)
    # =========================================================================
    nutrient_state_triples = [
        ("nutrient:magnesium", RelationType.ALLEVIATES, "state:stress", 0.95, "magnesium naturally lowers cortisol and alleviates stress"),
        ("nutrient:magnesium", RelationType.ALLEVIATES, "state:anxiety", 0.85, "magnesium soothes nervous tension and anxiety"),
        ("nutrient:omega-3", RelationType.REGULATES, "state:sad", 0.90, "omega-3 fats are linked to better mood regulation"),
        ("nutrient:omega-3", RelationType.ALLEVIATES, "state:stress", 0.80, "omega-3 fats are linked to better mood regulation"),
        ("nutrient:omega-3", RelationType.ALLEVIATES, "state:anxiety", 0.80, "omega-3 fats are linked to better mood regulation"),
        ("nutrient:complex-carb", RelationType.BOOSTS, "state:low_energy", 0.90, "complex carbohydrates give steady energy"),
        ("nutrient:complex-carb", RelationType.BOOSTS, "state:low_focus", 0.85, "complex carbohydrates give steady energy"),
        ("nutrient:complex-carb", RelationType.ALLEVIATES, "state:stress", 0.70, "complex carbohydrates give steady energy"),
        ("nutrient:complex-carb", RelationType.ALLEVIATES, "state:anxiety", 0.70, "complex carbohydrates give steady energy"),
        ("nutrient:high-protein", RelationType.BOOSTS, "state:low_focus", 0.90, "protein helps keep energy and focus stable"),
        ("nutrient:high-protein", RelationType.BOOSTS, "state:low_energy", 0.80, "protein helps keep energy and focus stable"),
        ("nutrient:high-protein", RelationType.REGULATES, "state:sad", 0.70, "protein helps keep energy and focus stable"),
        ("nutrient:antioxidants", RelationType.ALLEVIATES, "state:fatigue", 0.90, "antioxidants reduce oxidative stress causing mental fatigue"),
        ("nutrient:tryptophan", RelationType.ALLEVIATES, "state:anxiety", 0.85, "tryptophan boosts serotonin to alleviate anxiety"),
        ("nutrient:low-GI", RelationType.BOOSTS, "state:low_focus", 0.80, "low glycaemic index avoids energy crashes"),
    ]
    for h, r, t, w, exp in nutrient_state_triples:
        g.add_edge(h, t, relation=r, weight=w, explanation=exp)

    # Health Condition Contraindications
    g.add_edge("nutrient:high-GI", "condition:diabetes", relation=RelationType.CONTRAINDICATED_FOR, weight=1.0, explanation="high glycaemic index causes sharp glucose spikes in diabetes")
    g.add_edge("nutrient:high-sodium", "condition:hypertension", relation=RelationType.CONTRAINDICATED_FOR, weight=1.0, explanation="high sodium elevates blood pressure in hypertension")

    # =========================================================================
    # 6. FOOD ITEMS (NODES & TRIPLES) - Ingested from catalog
    # =========================================================================
    source_foods = foods_list if foods_list is not None else FOODS
    usda_map = load_usda_nutrients_map()

    for f in source_foods:
        fid = f"food:{f.name.lower().replace(' ', '_').replace('(', '').replace(')', '')}"
        food_name_lower = f.name.lower()
        matched_usda = {}
        for kw, nutr in usda_map.items():
            if kw in food_name_lower:
                matched_usda = nutr
                break

        props = {
            "calories": f.calories,
            "portion": f.portion,
            "meals": [m.value for m in f.meals],
            "tags": list(f.tags),
        }
        if matched_usda:
            props["usda_nutrients"] = matched_usda

        g.add_node(
            fid,
            node=KGNode(
                id=fid,
                name=f.name,
                node_type=NodeType.FOOD,
                properties=props,
            ),
        )

        for tag in f.tags:
            # Check if diet tag
            if tag in KNOWN_DIETS:
                g.add_edge(fid, f"diet:{tag}", relation=RelationType.COMPLIES_WITH, weight=1.0)
            # Check if allergen tag
            elif tag in KNOWN_ALLERGENS:
                g.add_edge(fid, f"allergen:{tag}", relation=RelationType.HAS_ALLERGEN, weight=1.0)
            # Check if nutrient tag
            elif tag in KNOWN_NUTRIENTS:
                w = 1.0
                if tag == "magnesium" and matched_usda.get("magnesium_mg", 0) > 60:
                    w = 1.2
                elif tag == "high-protein" and matched_usda.get("protein_g", 0) > 15:
                    w = 1.2
                g.add_edge(fid, f"nutrient:{tag}", relation=RelationType.CONTAINS, weight=w)

    # =========================================================================
    # 7. WELLNESS & PRODUCTIVITY TASKS (NODES & TRIPLES)
    # =========================================================================
    tasks_data = [
        (
            "task:breathing_478",
            "4-7-8 Parasympathetic Breathing",
            5, "personal",
            ["state:stress", "state:anxiety"],
            "A short breathing exercise helps bring acute stress down.",
        ),
        (
            "task:box_breathing",
            "Box Breathing Reset (4-4-4-4)",
            4, "personal",
            ["state:stress", "state:low_focus"],
            "Box breathing stabilizes autonomic tone and restores focus.",
        ),
        (
            "task:walk_outside",
            "10-minute walk outside",
            10, "exercise",
            ["state:stress", "state:anxiety", "state:fatigue"],
            "A short walk can help bring stress down.",
        ),
        (
            "task:rest_break",
            "20-minute rest break",
            20, "personal",
            ["state:fatigue", "state:low_energy"],
            "Mental fatigue is forecast to be high.",
        ),
        (
            "task:planning_break",
            "15-minute planning break",
            15, "personal",
            ["state:stress"],
            "Your schedule looks overloaded — re-prioritise.",
        ),
        (
            "task:hydration_pause",
            "Mindful Hydration & Posture Reset",
            3, "wellness",
            ["state:fatigue", "state:low_focus"],
            "Hydration break restores cerebral blood flow and focus.",
        ),
    ]

    for tid, name, dur_m, cat, states_alleviated, rationale in tasks_data:
        g.add_node(
            tid,
            node=KGNode(
                id=tid,
                name=name,
                node_type=NodeType.WELLNESS_TASK,
                properties={
                    "duration_minutes": dur_m,
                    "category": cat,
                    "rationale": rationale,
                },
            ),
        )
        for st in states_alleviated:
            g.add_edge(tid, st, relation=RelationType.ALLEVIATES, weight=1.0, explanation=rationale)

    return g
