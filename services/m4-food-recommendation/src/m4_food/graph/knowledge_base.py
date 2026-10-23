"""Layer B: Scientific Knowledge Graph Engine with Peer-Reviewed Provenance.

Builds a heterogeneous NetworkX DiGraph representing:
- Biological nutrients and their peer-reviewed relationships to mental/cognitive states
- Scientific evidence metadata (PMID, PMCID, DOI, Evidence Quality, Review Status)
- Deterministic medical contraindications (Diabetes, Hypertension)
- Verified culinary meal nodes with USDA and Sri Lankan Food Composition provenance
- Wellness micro-break interventions with clinical respiratory evidence

By default, edge weights between nutrients and states are set to EQUAL BASELINE (1.0).
Heuristic evidence-tier weighting (0.70 - 0.95) is available as an optional experimental strategy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import sys

src_dir = str(Path(__file__).resolve().parents[2])
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from m4_food.foods import FOODS, Food
from m4_food.graph.schema import KGNode, NodeType, RelationType

DATA_DIR = Path(__file__).resolve().parent / "data"
USDA_JSON_PATH = DATA_DIR / "usda_micronutrients.json"
HEARTSTEPS_JSON_PATH = DATA_DIR / "heartsteps_activities.json"


def load_heartsteps_activities() -> list[dict]:
    """Loads empirically verified JITAI wellness activity suggestions from HeartSteps MRT dataset."""
    if HEARTSTEPS_JSON_PATH.exists():
        try:
            with open(HEARTSTEPS_JSON_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


@dataclass(frozen=True)
class ScientificTriple:
    """Scientific evidence model for nutrient-state and task-state relationships."""
    head: str
    relation: RelationType
    tail: str
    scientific_description: str
    pmid: str | None = None
    pmcid: str | None = None
    doi: str | None = None
    evidence_level: str = "Peer-Reviewed Literature"
    review_status: str = "verified_literature"
    heuristic_tier_weight: float = 1.0  # Optional heuristic tier weight for ablation experiments


# =========================================================================
# SCIENTIFIC NUTRIENT-STATE RELATIONSHIPS (PEER-REVIEWED LITERATURE)
# =========================================================================
SCIENTIFIC_TRIPLES: list[ScientificTriple] = [
    ScientificTriple(
        head="nutrient:magnesium",
        relation=RelationType.ALLEVIATES,
        tail="state:stress",
        scientific_description="Magnesium modulates the hypothalamic-pituitary-adrenal (HPA) axis and acts as an NMDA receptor blocker. In systematic reviews, magnesium supplementation supports autonomic stress reduction.",
        pmid="28445426",
        pmcid="PMC5452159",
        doi="10.3390/nu9050429",
        evidence_level="Human Systematic Review & Preclinical In-Vivo",
        heuristic_tier_weight=0.95,
    ),
    ScientificTriple(
        head="nutrient:magnesium",
        relation=RelationType.ALLEVIATES,
        tail="state:anxiety",
        scientific_description="Magnesium deficiency is shown to induce anxiety-like behavior and neuronal hyper-excitability; dietary repletion supports central nervous system GABAergic transmission.",
        pmid="21835188",
        doi="10.1016/j.neuropharm.2011.07.027",
        evidence_level="Preclinical Neuropharmacology Trial",
        heuristic_tier_weight=0.85,
    ),
    ScientificTriple(
        head="nutrient:omega-3",
        relation=RelationType.REGULATES,
        tail="state:sad",
        scientific_description="Omega-3 polyunsaturated fatty acids (EPA/DHA) integrate into neuronal cell membranes, regulating neuroinflammation and monoaminergic neurotransmission in mood disorders.",
        pmid="24805797",
        pmcid="PMC4013159",
        doi="10.1371/journal.pone.0096905",
        evidence_level="Meta-Analysis of Randomized Controlled Trials",
        heuristic_tier_weight=0.90,
    ),
    ScientificTriple(
        head="nutrient:omega-3",
        relation=RelationType.ALLEVIATES,
        tail="state:stress",
        scientific_description="Omega-3 fatty acids attenuate elevated adrenal cortisol responses during sustained psychological and physical stress.",
        pmid="18568016",
        pmcid="PMC2805706",
        doi="10.1038/nrn2421",
        evidence_level="Neuroscience Review (Nature Reviews)",
        heuristic_tier_weight=0.80,
    ),
    ScientificTriple(
        head="nutrient:omega-3",
        relation=RelationType.ALLEVIATES,
        tail="state:anxiety",
        scientific_description="High-EPA formulations demonstrate anti-inflammatory neuroprotection, supporting anxiolytic physiological tone.",
        pmid="24805797",
        pmcid="PMC4013159",
        doi="10.1371/journal.pone.0096905",
        evidence_level="Meta-Analysis of Clinical Trials",
        heuristic_tier_weight=0.80,
    ),
    ScientificTriple(
        head="nutrient:complex-carb",
        relation=RelationType.BOOSTS,
        tail="state:low_energy",
        scientific_description="Complex carbohydrates undergo gradual enzymatic hydrolysis, providing steady glucose release into systemic circulation.",
        evidence_level="Metabolic Physiology / Glycemic Dynamics",
        heuristic_tier_weight=0.90,
    ),
    ScientificTriple(
        head="nutrient:complex-carb",
        relation=RelationType.BOOSTS,
        tail="state:low_focus",
        scientific_description="Dietary carbohydrates supply circulating blood glucose, the primary metabolic substrate utilized for cerebral cellular metabolism.",
        evidence_level="Cognitive Metabolism Literature",
        heuristic_tier_weight=0.85,
    ),
    ScientificTriple(
        head="nutrient:complex-carb",
        relation=RelationType.ALLEVIATES,
        tail="state:stress",
        scientific_description="Complex carbohydrates provide gradual systemic glucose release without rapid postprandial fluctuations.",
        evidence_level="Metabolic Endocrine Plausibility",
        heuristic_tier_weight=0.70,
    ),
    ScientificTriple(
        head="nutrient:complex-carb",
        relation=RelationType.ALLEVIATES,
        tail="state:anxiety",
        scientific_description="Sustained glycemic availability supports baseline metabolic stability during periods of elevated demand.",
        evidence_level="Metabolic Endocrine Plausibility",
        heuristic_tier_weight=0.70,
    ),
    ScientificTriple(
        head="nutrient:high-protein",
        relation=RelationType.BOOSTS,
        tail="state:low_focus",
        scientific_description="Dietary protein supplies amino acids including tyrosine, a dietary precursor involved in catecholamine neurotransmitter synthesis.",
        pmid="18568016",
        pmcid="PMC2805706",
        doi="10.1038/nrn2421",
        evidence_level="Neurochemistry Review",
        heuristic_tier_weight=0.90,
    ),
    ScientificTriple(
        head="nutrient:high-protein",
        relation=RelationType.BOOSTS,
        tail="state:low_energy",
        scientific_description="Dietary protein stimulates satiety pathways and contributes amino acid substrates for sustained metabolic turnover.",
        pmid="18568016",
        pmcid="PMC2805706",
        doi="10.1038/nrn2421",
        evidence_level="Neurochemistry Review",
        heuristic_tier_weight=0.80,
    ),
    ScientificTriple(
        head="nutrient:high-protein",
        relation=RelationType.REGULATES,
        tail="state:sad",
        scientific_description="Adequate complete amino acid profiles support steady neuroplasticity and monoaminergic balance.",
        pmid="18568016",
        pmcid="PMC2805706",
        doi="10.1038/nrn2421",
        evidence_level="Biological Plausibility",
        heuristic_tier_weight=0.70,
    ),
    ScientificTriple(
        head="nutrient:antioxidants",
        relation=RelationType.ALLEVIATES,
        tail="state:fatigue",
        scientific_description="Dietary polyphenols and antioxidants scavenge reactive oxygen species in cellular pathways.",
        pmid="18568016",
        pmcid="PMC2805706",
        doi="10.1038/nrn2421",
        evidence_level="Molecular Neuroscience Review",
        heuristic_tier_weight=0.90,
    ),
    ScientificTriple(
        head="nutrient:tryptophan",
        relation=RelationType.ALLEVIATES,
        tail="state:anxiety",
        scientific_description="L-Tryptophan is the direct, rate-limiting amino acid precursor for cerebral serotonin (5-HT) synthesis, regulating anxiety and mood homeostasis.",
        pmid="20651948",
        pmcid="PMC2908021",
        doi="10.4137/ijtr.s2129",
        evidence_level="Biochemical Review & Human Clinical Trials",
        heuristic_tier_weight=0.85,
    ),
    ScientificTriple(
        head="nutrient:low-GI",
        relation=RelationType.BOOSTS,
        tail="state:low_focus",
        scientific_description="Lower glycemic index carbohydrate sources are digested and absorbed more gradually than refined high-GI starches.",
        evidence_level="Clinical Nutrition Consensus",
        heuristic_tier_weight=0.80,
    ),
]

# Clinical Contraindication Guidelines
CLINICAL_CONTRAINDICATIONS = [
    {
        "head": "nutrient:high-GI",
        "tail": "condition:diabetes",
        "relation": RelationType.CONTRAINDICATED_FOR,
        "explanation": "High glycemic index starches cause acute postprandial glucose surges in Type-2 Diabetes.",
        "guideline_body": "American Diabetes Association (ADA) Standards of Care 2024",
    },
    {
        "head": "nutrient:high-sodium",
        "tail": "condition:hypertension",
        "relation": RelationType.CONTRAINDICATED_FOR,
        "explanation": "Excessive dietary sodium elevates extracellular volume and vascular resistance in Hypertension.",
        "guideline_body": "American Heart Association (AHA) Dietary Sodium Guidelines",
    },
]

# Wellness Micro-Intervention Evidence
TASK_EVIDENCE = [
    {
        "task_id": "task:box_breathing",
        "title": "Box Breathing Reset (4-4-4-4)",
        "duration": 4,
        "category": "personal",
        "states": ["state:stress", "state:low_focus"],
        "explanation": "Equalized 4-second box respiration enhances autonomic tone and restores vagal cognitive control.",
        "pmid": "36630953",
        "pmcid": "PMC9873947",
        "source": "Stanford University RCT (Cell Reports Medicine 2023)",
    },
    {
        "task_id": "task:breathing_478",
        "title": "4-7-8 Parasympathetic Breathing",
        "duration": 5,
        "category": "personal",
        "states": ["state:stress", "state:anxiety"],
        "explanation": "Prolonged 8-second exhalation triggers parasympathetic vagal activation, rapidly lowering acute anxiety.",
        "pmid": "36630953",
        "pmcid": "PMC9873947",
        "source": "Clinical Respiratory Physiology Protocol",
    },
    {
        "task_id": "task:walk_outside",
        "title": "10-minute walk outside",
        "duration": 10,
        "category": "exercise",
        "states": ["state:stress", "state:anxiety", "state:fatigue"],
        "explanation": "Brief outdoor aerobic movement reduces sympathetic cortisol and mitigates mental fatigue.",
        "pmid": "31464168",
        "source": "Frontiers in Public Health (Urban Nature Micro-breaks)",
    },
    {
        "task_id": "task:rest_break",
        "title": "20-minute rest break",
        "duration": 20,
        "category": "personal",
        "states": ["state:fatigue", "state:low_energy"],
        "explanation": "Non-sleep deep rest restores prefrontal metabolic reserves during high cognitive exhaustion.",
        "pmid": "34509121",
        "source": "Occupational Ergonomics Literature",
    },
    {
        "task_id": "task:planning_break",
        "title": "15-minute planning break",
        "duration": 15,
        "category": "work",
        "states": ["state:stress"],
        "explanation": "Cognitive re-prioritization pause reduces schedule overload and subjective stress.",
        "source": "Ergonomics Task Management Protocol",
    },
    {
        "task_id": "task:hydration_pause",
        "title": "Mindful Hydration & Posture Reset",
        "duration": 3,
        "category": "personal",
        "states": ["state:fatigue", "state:low_focus"],
        "explanation": "Rehydration restores cerebral perfusion and counters mild dehydration-induced lethargy.",
        "pmid": "22190027",
        "source": "Journal of Nutrition (Dehydration & Cognition)",
    },
]

KNOWN_ALLERGENS = frozenset({"dairy", "egg", "fish", "gluten", "nuts", "soy", "shellfish", "peanut", "coconut"})
KNOWN_DIETS = frozenset({"vegan", "vegetarian", "halal"})
KNOWN_NUTRIENTS = frozenset({
    "magnesium", "omega-3", "complex-carb", "high-protein", "antioxidants",
    "tryptophan", "potassium", "low-GI", "high-GI", "high-sodium", "high-sugar",
    "fried", "caffeine",
})


def load_usda_nutrients_map() -> dict[str, dict[str, float]]:
    if not USDA_JSON_PATH.exists():
        return {}
    try:
        with open(USDA_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {item.get("keyword", ""): item.get("nutrients", {}) for item in data.values()}
    except Exception:
        return {}


def create_default_knowledge_graph(
    foods_list: list[Food] | None = None,
    use_heuristic_weights: bool = False,
) -> nx.DiGraph:
    """Builds the scientific Knowledge Graph with explicit data provenance.
    
    Args:
        foods_list: Custom food catalog or defaults to Layer A FOODS.
        use_heuristic_weights: If False (default), all therapeutic edges use equal 1.0 baseline weight.
                               If True, uses the optional evidence-tier weights (0.70 - 0.95) for ablation.
    """
    g = nx.DiGraph()

    # 1. Affective & Cognitive States
    states = [
        ("state:stress", "Stress / Tension", NodeType.AFFECTIVE_STATE),
        ("state:anxiety", "Anxiety", NodeType.AFFECTIVE_STATE),
        ("state:sad", "Low Mood / Sadness", NodeType.AFFECTIVE_STATE),
        ("state:fatigue", "Mental Fatigue / Exhaustion", NodeType.COGNITIVE_STATE),
        ("state:low_energy", "Low Energy / Lethargy", NodeType.COGNITIVE_STATE),
        ("state:low_focus", "Poor Focus / Brain Fog", NodeType.COGNITIVE_STATE),
    ]
    for sid, name, ntype in states:
        g.add_node(sid, node=KGNode(id=sid, name=name, node_type=ntype))

    # 2. Nutrients
    nutrients = [
        ("nutrient:magnesium", "Magnesium"),
        ("nutrient:omega-3", "Omega-3 Fatty Acids"),
        ("nutrient:complex-carb", "Complex Carbohydrates"),
        ("nutrient:high-protein", "Quality Protein & Amino Acids"),
        ("nutrient:antioxidants", "Polyphenols & Antioxidants"),
        ("nutrient:tryptophan", "L-Tryptophan"),
        ("nutrient:potassium", "Potassium"),
        ("nutrient:low-GI", "Low Glycemic Index"),
        ("nutrient:high-GI", "High Glycemic Index"),
        ("nutrient:high-sodium", "High Sodium"),
        ("nutrient:high-sugar", "High Sugar"),
        ("nutrient:fried", "Deep Fried / High Saturated Fat"),
        ("nutrient:caffeine", "Caffeine"),
    ]
    for nid, name in nutrients:
        g.add_node(nid, node=KGNode(id=nid, name=name, node_type=NodeType.NUTRIENT))

    # 3. Clinical Conditions, Allergens & Diets
    conditions = [
        ("condition:diabetes", "Diabetes", NodeType.HEALTH_CONDITION),
        ("condition:hypertension", "Hypertension", NodeType.HEALTH_CONDITION),
    ]
    for cid, name, ntype in conditions:
        g.add_node(cid, node=KGNode(id=cid, name=name, node_type=ntype))

    for alg in KNOWN_ALLERGENS:
        aid = f"allergen:{alg}"
        g.add_node(aid, node=KGNode(id=aid, name=alg.capitalize(), node_type=NodeType.ALLERGEN))

    for diet in KNOWN_DIETS:
        did = f"diet:{diet}"
        g.add_node(did, node=KGNode(id=did, name=diet.capitalize(), node_type=NodeType.DIETARY_PREFERENCE))

    # 4. Wellness Micro-Tasks with Clinical Evidence
    for task in TASK_EVIDENCE:
        tid = task["task_id"]
        g.add_node(
            tid,
            node=KGNode(
                id=tid,
                name=task["title"],
                node_type=NodeType.WELLNESS_TASK,
                properties={
                    "duration_minutes": task["duration"],
                    "category": task["category"],
                    "pmid": task.get("pmid"),
                    "source": task.get("source"),
                },
            ),
        )
        for st in task["states"]:
            g.add_edge(
                tid,
                st,
                relation=RelationType.ALLEVIATES,
                weight=1.0,
                explanation=task["explanation"],
                pmid=task.get("pmid"),
            )

    # 4b. HeartSteps Micro-Randomized Trial (MRT) Empirical Activity Tasks
    heartsteps_tasks = load_heartsteps_activities()
    for ht in heartsteps_tasks:
        tid = ht["task_id"]
        g.add_node(
            tid,
            node=KGNode(
                id=tid,
                name=ht["title"],
                node_type=NodeType.WELLNESS_TASK,
                properties={
                    "duration_minutes": ht["duration"],
                    "category": ht["category"],
                    "message": ht["message"],
                    "explanation": ht["explanation"],
                    "source": ht["source"],
                    "empirical_stats": ht.get("empirical_stats", {}),
                },
            ),
        )
        pos_ratio = ht.get("empirical_stats", {}).get("positive_ratio", 0.8)
        for st in ht["states"]:
            g.add_edge(
                tid,
                st,
                relation=RelationType.ALLEVIATES,
                weight=round(pos_ratio, 2),
                explanation=ht["explanation"],
                source=ht["source"],
            )

    # 5. Scientific Nutrient-State Triples (Equal 1.0 Baseline or Optional Heuristic)
    for triple in SCIENTIFIC_TRIPLES:
        edge_weight = triple.heuristic_tier_weight if use_heuristic_weights else 1.0
        g.add_edge(
            triple.head,
            triple.tail,
            relation=triple.relation,
            weight=edge_weight,
            explanation=triple.scientific_description,
            pmid=triple.pmid,
            pmcid=triple.pmcid,
            doi=triple.doi,
            evidence_level=triple.evidence_level,
            review_status=triple.review_status,
            heuristic_tier_weight=triple.heuristic_tier_weight,
        )

    # Clinical Contraindications (Deterministic 1.0 Hard Constraint)
    for cc in CLINICAL_CONTRAINDICATIONS:
        g.add_edge(
            cc["head"],
            cc["tail"],
            relation=cc["relation"],
            weight=1.0,
            explanation=cc["explanation"],
            guideline=cc["guideline_body"],
        )

    # 6. Food Ingestion (Layer A with Provenance Tracking)
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
            "food_id": f.food_id,
            "source_db": f.source_db,
            "source_id": f.source_id,
            "measurement_basis": f.measurement_basis,
            "provenance_notes": f.provenance_notes,
        }
        if matched_usda:
            props["usda_nutrients"] = matched_usda

        # Add verified nutrient attributes if present
        if f.protein_g is not None: props["protein_g"] = f.protein_g
        if f.magnesium_mg is not None: props["magnesium_mg"] = f.magnesium_mg
        if f.potassium_mg is not None: props["potassium_mg"] = f.potassium_mg
        if f.polyunsaturated_fat_g is not None: props["polyunsaturated_fat_g"] = f.polyunsaturated_fat_g

        g.add_node(
            fid,
            node=KGNode(
                id=fid,
                name=f.name,
                node_type=NodeType.FOOD,
                properties=props,
            ),
        )

        # Diets, Allergens, and Nutrients
        if f.is_vegan or "vegan" in f.tags:
            g.add_edge(fid, "diet:vegan", relation=RelationType.COMPLIES_WITH, weight=1.0)
        if f.is_vegetarian or "vegetarian" in f.tags:
            g.add_edge(fid, "diet:vegetarian", relation=RelationType.COMPLIES_WITH, weight=1.0)
        if f.is_halal or "halal" in f.tags:
            g.add_edge(fid, "diet:halal", relation=RelationType.COMPLIES_WITH, weight=1.0)

        all_food_allergens = set(f.tags) | set(getattr(f, "allergens", ())) | set(getattr(f, "fda_allergens", ()))
        for ing in getattr(f, "ingredients", ()):
            all_food_allergens.update(getattr(ing, "allergens", ()))

        for alg in all_food_allergens:
            if alg in KNOWN_ALLERGENS:
                g.add_edge(fid, f"allergen:{alg}", relation=RelationType.HAS_ALLERGEN, weight=1.0)

        for tag in f.tags:
            if tag in KNOWN_NUTRIENTS:
                w = 1.0
                if use_heuristic_weights:
                    if tag == "magnesium" and matched_usda.get("magnesium_mg", 0) > 60:
                        w = 1.2
                    elif tag == "high-protein" and matched_usda.get("protein_g", 0) > 15:
                        w = 1.2
                g.add_edge(fid, f"nutrient:{tag}", relation=RelationType.CONTAINS, weight=w)

    return g


if __name__ == "__main__":
    g_baseline = create_default_knowledge_graph(use_heuristic_weights=False)
    print(f"Scientific Knowledge Graph built with Equal 1.0 Baseline: {len(g_baseline.nodes())} nodes, {len(g_baseline.edges())} edges.")
    sample_edge = g_baseline.get_edge_data("nutrient:magnesium", "state:stress")
    print(f"Sample Edge (Magnesium -> Stress): Weight={sample_edge.get('weight')}, PMID={sample_edge.get('pmid')}")
