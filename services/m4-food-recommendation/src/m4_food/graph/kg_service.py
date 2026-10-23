"""Knowledge Graph Service Adapter for the FastAPI Endpoint.

Translates wellness_contracts.RecommendationRequest into Knowledge Graph traversal,
safety filtering, context scoring, and returns MealRecommendation and SuggestedActivity contracts.

Enhancements:
- Canonical allergen normalization supporting FDA major allergens + culinary aliases (dairy, tree nuts, etc.)
- Strict separation of coconut from FDA tree nuts, and gluten from wheat
- Unresolved allergen review status handling for allergy-sensitive users
- Configurable clinical safety thresholds for Diabetes (Glycemic Load/Index) and Hypertension (Sodium mg)
- Active state isolation (no double counting of stress and anxiety)
- Verified nutrient telemetry thresholds with provenance tracking
- Configurable scoring weights and equal-weight baseline for ablation evaluation
- Knowledge Graph and GNN-driven dynamic SuggestedActivity generation from graph task nodes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo
import networkx as nx

try:
    from m4_food.foods import FOODS, Food
    from m4_food.recommender import (
        CALORIE_SHARE,
        DEFAULT_DAILY_KCAL,
        ITEMS_PER_MEAL,
        VARIETY_PENALTY,
        busy_intervals,
        energy_at,
        meal_windows,
    )
except ImportError:
    from ..foods import FOODS, Food
    from ..recommender import (
        CALORIE_SHARE,
        DEFAULT_DAILY_KCAL,
        ITEMS_PER_MEAL,
        VARIETY_PENALTY,
        busy_intervals,
        energy_at,
        meal_windows,
    )
from wellness_contracts import (
    AffectiveLabel,
    DietProfile,
    FoodItem,
    MealRecommendation,
    MealType,
    RecommendationRequest,
    SuggestedActivity,
    TaskCategory,
    WellnessForecast,
)

from .gnn_ranker import GNNRecommenderRanker
from .knowledge_base import KNOWN_ALLERGENS, KNOWN_DIETS, create_default_knowledge_graph
from .schema import KGNode, NodeType, RelationType


# =========================================================================
# CANONICAL ALLERGEN MAPPING & ALIAS DICTIONARY
# =========================================================================
CANONICAL_ALLERGEN_ALIASES: dict[str, str] = {
    # Dairy / Milk
    "dairy": "milk",
    "milk": "milk",
    "cow_milk": "milk",
    "cow milk": "milk",
    "lactose": "milk",
    "cheese": "milk",
    "curd": "milk",
    "yoghurt": "milk",
    "yogurt": "milk",
    "butter": "milk",
    "ghee": "milk",
    "whey": "milk",
    "casein": "milk",
    # Eggs
    "egg": "eggs",
    "eggs": "eggs",
    "egg_white": "eggs",
    "yolk": "eggs",
    "albumen": "eggs",
    # Fish
    "fish": "fish",
    "finfish": "fish",
    "salmon": "fish",
    "tuna": "fish",
    "mackerel": "fish",
    "cod": "fish",
    "tilapia": "fish",
    # Crustacean Shellfish
    "shellfish": "crustacean_shellfish",
    "crustacean": "crustacean_shellfish",
    "crustaceans": "crustacean_shellfish",
    "crustacean_shellfish": "crustacean_shellfish",
    "prawn": "crustacean_shellfish",
    "prawns": "crustacean_shellfish",
    "shrimp": "crustacean_shellfish",
    "shrimps": "crustacean_shellfish",
    "crab": "crustacean_shellfish",
    "lobster": "crustacean_shellfish",
    # Peanuts
    "peanut": "peanuts",
    "peanuts": "peanuts",
    "groundnut": "peanuts",
    "groundnuts": "peanuts",
    "arachis": "peanuts",
    # Tree Nuts (Strictly excluding coconut!)
    "nuts": "tree_nuts",
    "tree_nuts": "tree_nuts",
    "tree nut": "tree_nuts",
    "tree nuts": "tree_nuts",
    "treenuts": "tree_nuts",
    "almond": "tree_nuts",
    "almonds": "tree_nuts",
    "walnut": "tree_nuts",
    "walnuts": "tree_nuts",
    "cashew": "tree_nuts",
    "cashews": "tree_nuts",
    "pistachio": "tree_nuts",
    "pistachios": "tree_nuts",
    "pecan": "tree_nuts",
    "pecans": "tree_nuts",
    "hazelnut": "tree_nuts",
    "hazelnuts": "tree_nuts",
    # Wheat
    "wheat": "wheat",
    "wheat flour": "wheat",
    "atta": "wheat",
    "semolina": "wheat",
    # Gluten (Dietary celiac tag, strictly distinguished from wheat allergy alone)
    "gluten": "gluten",
    "celiac": "gluten",
    "coeliac": "gluten",
    # Soybeans
    "soy": "soybeans",
    "soya": "soybeans",
    "soybean": "soybeans",
    "soybeans": "soybeans",
    "tofu": "soybeans",
    "edamame": "soybeans",
    # Sesame
    "sesame": "sesame",
    "sesame_seed": "sesame",
    "sesame_seeds": "sesame",
    "sesame seed": "sesame",
    "sesame seeds": "sesame",
    "gingelly": "sesame",
    "til": "sesame",
    "tahini": "sesame",
    # Coconut (Strictly tracked separately from FDA tree nuts)
    "coconut": "coconut",
    "pol": "coconut",
    "thambili": "coconut",
}


def normalize_allergen(raw: str) -> str:
    """Normalizes an allergen string to its canonical identifier."""
    cleaned = raw.lower().strip().replace("-", "_")
    return CANONICAL_ALLERGEN_ALIASES.get(cleaned, cleaned)


def extract_food_allergens(food: Food, g: nx.DiGraph | None = None, fid: str | None = None) -> set[str]:
    """Extracts and normalizes all declared allergens across structured attributes, ingredients, tags, and KG edges."""
    declared = set()
    for a in getattr(food, "allergens", ()):
        declared.add(normalize_allergen(a))
    for a in getattr(food, "fda_allergens", ()):
        declared.add(normalize_allergen(a))
    for ing in getattr(food, "ingredients", ()):
        for a in getattr(ing, "allergens", ()):
            declared.add(normalize_allergen(a))
    for t in getattr(food, "tags", ()):
        norm = normalize_allergen(t)
        if norm in CANONICAL_ALLERGEN_ALIASES.values():
            declared.add(norm)
    if g is not None and fid is not None and g.has_node(fid):
        for _, tail, data in g.out_edges(fid, data=True):
            if data.get("relation") == RelationType.HAS_ALLERGEN:
                tag = tail.replace("allergen:", "")
                declared.add(normalize_allergen(tag))
    # Biological Fact: Wheat grain contains gluten protein
    if "wheat" in declared:
        declared.add("gluten")
    return declared


# =========================================================================
# CONFIGURABLE SCORING WEIGHTS & CLINICAL THRESHOLDS
# =========================================================================
@dataclass(frozen=True)
class ScoringWeights:
    """Configurable coefficients for ranking and heuristic evaluation."""
    kg_path_weight: float = 1.0           # Weight for KG nutrient-state path traversals
    gnn_weight: float = 0.3               # Weight for LightGCN cosine similarity
    verified_nutrient_weight: float = 0.5 # Boost for laboratory/calculated nutrient threshold matches
    caffeine_penalty: float = 1.0         # Penalty for caffeine during stress/anxiety
    late_caffeine_penalty: float = 2.0    # Penalty for caffeine after afternoon threshold
    fried_penalty: float = 0.6            # Penalty for deep fried items
    high_sugar_penalty: float = 0.5       # Penalty for high sugar items
    high_sodium_penalty: float = 0.4      # Penalty for elevated sodium
    high_gi_penalty: float = 0.2          # Penalty for high GI during energy dips
    low_gi_boost: float = 0.3             # Boost for low GI sustained glucose release
    calorie_proximity_weight: float = 0.6 # Penalty for deviation from target calories
    variety_penalty: float = 0.5          # Penalty for same-day repeated items


EQUAL_WEIGHT_BASELINE = ScoringWeights(
    kg_path_weight=1.0,
    gnn_weight=0.0,
    verified_nutrient_weight=0.0,
    caffeine_penalty=0.0,
    late_caffeine_penalty=0.0,
    fried_penalty=0.0,
    high_sugar_penalty=0.0,
    high_sodium_penalty=0.0,
    high_gi_penalty=0.0,
    low_gi_boost=0.0,
    calorie_proximity_weight=0.0,
    variety_penalty=0.0,
)


@dataclass(frozen=True)
class ClinicalSafetyThresholds:
    """Clinical thresholds for medical contraindication pruning and completeness validation."""
    diabetes_max_gi: int = 70                # ADA threshold for high GI carbohydrates
    diabetes_max_gl: float = 20.0            # High Glycemic Load threshold
    hypertension_max_sodium_mg: float = 400.0# AHA single-meal sodium limit (approx. 20% of 2300mg DV)
    affective_state_expiry_hours: float = 6.0# Configurable heuristic validity horizon for acute affective observations
    min_meal_calorie_ratio: float = 0.70     # Minimum acceptable proportion of meal calorie allocation (main meals)
    min_snack_calorie_ratio: float = 0.50    # Minimum acceptable proportion for snacks
    min_main_protein_g: float = 12.0         # Minimum protein baseline for complete main meals
    min_snack_protein_g: float = 4.0         # Minimum protein baseline for snacks


# =========================================================================
# KNOWLEDGE GRAPH SERVICE IMPLEMENTATION
# =========================================================================
class KnowledgeGraphService:
    def __init__(
        self,
        foods_list: list[Food] | None = None,
        enable_gnn: bool = True,
        weights: ScoringWeights | None = None,
        thresholds: ClinicalSafetyThresholds | None = None,
    ):
        self.foods = foods_list if foods_list is not None else FOODS
        self.g = create_default_knowledge_graph(self.foods)
        self.foods_by_name = {f.name: f for f in self.foods}
        self.gnn = GNNRecommenderRanker() if enable_gnn else None
        self.weights = weights if weights is not None else ScoringWeights()
        self.thresholds = thresholds if thresholds is not None else ClinicalSafetyThresholds()

    def _fid(self, name: str) -> str:
        return f"food:{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}"

    def is_safe_kg(self, food: Food, profile: DietProfile) -> bool:
        """Deterministic safety and compliance filter.
        
        Evaluates:
        1. Dietary preferences (vegan, vegetarian, halal) without inferring compliance from missing data.
        2. Strict multi-source allergen presence with canonical normalization and alias resolution.
        3. Excludes unresolved allergen review statuses and cross-contamination risks for allergy-sensitive users.
        4. User declared dislikes (including comprehensive detection of spicy curries, sambols, and seasonings).
        5. Clinical condition contraindications:
           - Diabetes via Glycemic Load (GL <= 20) and available carbohydrate limits.
           - Hypertension via Sodium mg (<= 400mg per meal).
        """
        fid = self._fid(food.name)
        if not self.g.has_node(fid):
            return False

        # 1. Deterministic Dietary Compliance
        if profile.vegan:
            if not (getattr(food, "is_vegan", False) or "vegan" in food.tags or self.g.has_edge(fid, "diet:vegan")):
                return False
        if profile.vegetarian:
            if not (getattr(food, "is_vegetarian", False) or "vegetarian" in food.tags or self.g.has_edge(fid, "diet:vegetarian")):
                return False
        if profile.halal:
            if not (getattr(food, "is_halal", False) or "halal" in food.tags or self.g.has_edge(fid, "diet:halal")):
                return False

        # 2. Allergens Check
        user_allergies = {normalize_allergen(a) for a in profile.allergies if a.strip()}
        if user_allergies:
            # Unresolved allergen review status or cross-contamination risk cannot be assumed safe for allergy-sensitive users
            review_status = getattr(food, "allergen_review_status", "")
            if review_status in ("UNVERIFIED_REVIEW_PENDING", "UNVERIFIED_CROSS_CONTAMINATION_RISK"):
                return False

            food_allergens = extract_food_allergens(food, self.g, fid)
            if user_allergies & food_allergens:
                return False

        # 3. Dislikes (with explicit spicy seasoning and curry detection)
        food_name_lower = food.name.lower()
        ing_names = " ".join(ing.name.lower() for ing in getattr(food, "ingredients", ()))
        for d in (x.lower().strip() for x in profile.dislikes):
            if not d:
                continue
            if "spic" in d:
                spicy_markers = ("spicy", "curry", "chili", "chilli", "pepper", "sambol", "miris", "devilled", "tempered")
                if "spicy" in food.tags or any(m in food_name_lower or m in ing_names for m in spicy_markers):
                    return False
            elif d in food_name_lower or d in food.tags or d in ing_names:
                return False

        # 4. Medical / Health Condition Contraindications
        conditions_str = " ".join(profile.health_conditions).lower()
        if "diabet" in conditions_str:
            is_negligible_carb = (
                getattr(food, "gi_method", "") == "NOT_APPLICABLE_NEGLIGIBLE_CARB"
                or (food.carbs_g is not None and food.carbs_g <= 3.0)
            )
            if not is_negligible_carb:
                gi = getattr(food, "glycemic_index", None)
                carbs = getattr(food, "carbs_g", None)
                fiber = getattr(food, "fiber_g", 0.0) or 0.0
                avail_carbs = max(0.0, carbs - fiber) if carbs is not None else None

                if gi is not None:
                    if avail_carbs is not None:
                        gl = (gi * avail_carbs) / 100.0
                        if gl >= self.thresholds.diabetes_max_gl or (gi >= self.thresholds.diabetes_max_gi and avail_carbs >= 15.0):
                            return False
                    elif gi >= self.thresholds.diabetes_max_gi:
                        return False
                else:
                    if "high-GI" in food.tags or "high-sugar" in food.tags or self.g.has_edge(fid, "nutrient:high-GI"):
                        return False
                    if carbs is not None and carbs > 30.0 and "low-GI" not in food.tags:
                        return False

        if "hypertension" in conditions_str or "blood pressure" in conditions_str:
            sodium_mg = getattr(food, "sodium_mg", None)
            if sodium_mg is not None:
                if sodium_mg > self.thresholds.hypertension_max_sodium_mg:
                    return False
            else:
                if "high-sodium" in food.tags or self.g.has_edge(fid, "nutrient:high-sodium"):
                    return False

        return True

    def score_food_kg(
        self,
        food: Food,
        meal: MealType,
        label: AffectiveLabel | None,
        energy: float | None,
        local_hour: int,
        kcal_target: float,
        wellness: WellnessForecast | None = None,
        profile: DietProfile | None = None,
    ) -> tuple[float, list[str]]:
        """Scores candidate food using Knowledge Graph path traversal and GNN representations.
        
        Isolates stress vs anxiety to avoid double counting, leverages verified nutrient measurements,
        and provides calibrated explainability without exaggerated medical claims.
        """
        fid = self._fid(food.name)
        tags = food.tags
        score = 0.0
        reasons = []

        # 1. Determine active context states (strictly without double counting stress and anxiety)
        active_states: list[str] = []
        if label == AffectiveLabel.stress:
            active_states.append("state:stress")
        elif label == AffectiveLabel.anxiety:
            active_states.append("state:anxiety")
        elif label == AffectiveLabel.sad:
            active_states.append("state:sad")
        elif label == AffectiveLabel.relaxed:
            active_states.append("state:relaxed")

        if energy is not None and energy < 0.4:
            active_states.append("state:low_energy")

        if wellness is not None and wellness.slots:
            next_slots = wellness.slots[:3]
            avg_fatigue = sum(s.mental_fatigue for s in next_slots) / len(next_slots)
            if avg_fatigue > 0.6:
                active_states.append("state:fatigue")
            avg_focus = sum(s.focus for s in next_slots) / len(next_slots)
            if avg_focus < 0.4:
                active_states.append("state:low_focus")

        # 2. Knowledge Graph Path Traversal: Food -> Nutrient -> Active States
        for _, nutrient_id, edge_data in self.g.out_edges(fid, data=True):
            if edge_data.get("relation") != RelationType.CONTAINS:
                continue

            for state_id in active_states:
                if self.g.has_edge(nutrient_id, state_id):
                    st_data = self.g.get_edge_data(nutrient_id, state_id)
                    w = float(st_data.get("weight", 1.0))
                    score += self.weights.kg_path_weight * w
                    exp = st_data.get("explanation", "")
                    pmid = st_data.get("pmid")
                    if pmid:
                        exp = f"{exp} (PMID: {pmid})"
                    if exp and exp not in reasons:
                        reasons.append(exp)

        # Determine actual source-provenance description
        rec_type = getattr(food, "record_type", "")
        prov_status = getattr(food, "data_provenance_status", "")
        if rec_type == "PRIMARY_LABORATORY_ASSAY":
            prov_desc = f"Lab Assay ({food.source_db})"
        elif rec_type == "CALCULATED_COMPOSITE":
            prov_desc = f"Recipe Calculation ({prov_status})"
        else:
            prov_desc = f"Regional Estimate ({prov_status})"

        # 3. Verified Nutrient Telemetry & Calibrated Explanations (Strictly no unproven curative claims)
        if food.magnesium_mg is not None:
            if food.magnesium_mg >= 60.0:
                score += self.weights.verified_nutrient_weight * 0.5
                if "state:stress" in active_states or "state:anxiety" in active_states:
                    reasons.append(f"[{prov_desc}] Provides {food.magnesium_mg}mg magnesium; essential cofactor in cellular energy metabolism (PMID: 28445426)")
            elif food.magnesium_mg >= 30.0:
                score += self.weights.verified_nutrient_weight * 0.25
        elif "magnesium" in tags:
            score += 0.2
            if "state:stress" in active_states or "state:anxiety" in active_states:
                reasons.append("Magnesium rich (culinary tag estimate)")

        if food.omega3_g is not None and food.omega3_g >= 0.25:
            score += self.weights.verified_nutrient_weight * 0.5
            if "state:sad" in active_states or "state:stress" in active_states:
                reasons.append(f"[{prov_desc}] Provides {food.omega3_g}g omega-3; structural neuronal membrane lipids (PMID: 24805797)")
        elif "omega-3" in tags and food.omega3_g is None:
            score += 0.2
            reasons.append("Omega-3 fatty acids (culinary tag estimate)")

        if food.protein_g is not None and food.protein_g >= 15.0:
            if "state:low_energy" in active_states or "state:low_focus" in active_states:
                score += self.weights.verified_nutrient_weight * 0.4
                reasons.append(f"[{prov_desc}] Provides {food.protein_g}g dietary protein; contributes amino acids for muscle maintenance and satiety (PMID: 18568016)")

        # 4. Contextual Nuances & Penalties
        if "caffeine" in tags:
            if label in (AffectiveLabel.stress, AffectiveLabel.anxiety):
                score -= self.weights.caffeine_penalty
                reasons.append("caffeine avoided during elevated stress/anxiety")
            if local_hour >= 15:
                score -= self.weights.late_caffeine_penalty
                reasons.append("caffeine avoided after 15:00 to protect sleep")

        if "fried" in tags:
            score -= self.weights.fried_penalty
        if "high-sugar" in tags:
            score -= self.weights.high_sugar_penalty
        if "high-sodium" in tags:
            score -= self.weights.high_sodium_penalty
        if "high-GI" in tags:
            score -= self.weights.high_gi_penalty

        # Diabetes Telemetry vs Standard Glycemic Explanations
        is_diabetes = False
        if profile and profile.health_conditions:
            conds_str = " ".join(profile.health_conditions).lower()
            if "diabetes" in conds_str:
                is_diabetes = True

        if is_diabetes:
            carbs = food.carbs_g
            fiber = food.fiber_g or 0.0
            if carbs is not None and carbs > 3.0:
                net_carbs = max(0.0, carbs - fiber)
                gi_str = f"GI: {food.glycemic_index}" if food.glycemic_index is not None else "GI: unlisted"
                gl_str = f", GL: {(food.glycemic_index * net_carbs)/100:.1f}" if food.glycemic_index is not None else ""
                reasons.append(
                    f"Portion: {food.portion} (total carbs: {carbs:.1f}g, fiber: {fiber:.1f}g, net carbs: {net_carbs:.1f}g{gl_str}; {gi_str}); supports gradual postprandial glucose absorption"
                )
            else:
                p_val = food.protein_g or 0.0
                reasons.append(
                    f"Portion: {food.portion} (negligible carbohydrate, protein: {p_val:.1f}g); minimal postprandial glycemic impact"
                )
        elif "low-GI" in tags or (food.glycemic_index is not None and food.glycemic_index <= 55):
            score += self.weights.low_gi_boost
            gi_val = food.glycemic_index if food.glycemic_index is not None else "low-GI tag"
            reasons.append(f"Low glycemic index ({gi_val}) supports gradual carbohydrate absorption")

        # 5. GNN Neural Representation Score (Cosine Similarity)
        gnn_score_val = 0.0
        if self.gnn is not None and self.gnn.is_loaded and self.gnn.final_embeddings is not None:
            f_idx = self.gnn.node_to_idx.get(fid)
            if f_idx is not None:
                for state_id in active_states:
                    s_idx = self.gnn.node_to_idx.get(state_id)
                    if s_idx is not None:
                        sim = float(self.gnn.model.cosine_score(self.gnn.final_embeddings, s_idx, f_idx))
                        gnn_score_val += sim
                        score += self.weights.gnn_weight * sim

        if gnn_score_val != 0.0:
            reasons.insert(
                0,
                f"[GNN Score (Structural Proximity): {gnn_score_val:+.2f} (graph link-prediction score; reflects knowledge graph topology, not clinical suitability or personalized preference)]"
            )

        # 6. Calorie Proximity
        score -= self.weights.calorie_proximity_weight * abs(food.calories - kcal_target) / max(kcal_target, 1)

        return score, reasons

    def pick_foods_kg(
        self,
        meal: MealType,
        label: AffectiveLabel | None,
        energy: float | None,
        local_hour: int,
        profile: DietProfile,
        used: dict[str, int],
        wellness: WellnessForecast | None = None,
    ) -> list[FoodItem]:
        daily = profile.daily_calorie_target or DEFAULT_DAILY_KCAL
        n = ITEMS_PER_MEAL[meal]
        target = daily * CALORIE_SHARE[meal] / n

        # Filter safe foods using deterministic Knowledge Graph constraints
        cands = [f for f in self.foods if meal in f.meals and self.is_safe_kg(f, profile)]
        if not cands:
            return []

        # Variety constraint: count unused safe candidates
        unused_cands = [f for f in cands if used.get(f.name, 0) == 0]

        scored = []
        for f in cands:
            sc, why = self.score_food_kg(f, meal, label, energy, local_hour, target, wellness, profile=profile)
            times_used = used.get(f.name, 0)
            # If food was already used and enough unused alternative options exist, penalize strongly to avoid repeated meals
            if times_used > 0 and len(unused_cands) >= n:
                penalty = 15.0 * times_used
            else:
                penalty = self.weights.variety_penalty * times_used
            scored.append(((sc - penalty, why), f))

        ranked = sorted(scored, key=lambda x: (-x[0][0], x[1].name))
        out = []
        for (_, why), f in ranked[:n]:
            used[f.name] = used.get(f.name, 0) + 1
            # Compact and deduplicate explanation parts
            seen_why = set()
            compact_why = []
            for w in why:
                if w not in seen_why:
                    seen_why.add(w)
                    compact_why.append(w)
            reason = "; ".join(compact_why[:3]) if compact_why else "fits your profile and the time of day"
            out.append(FoodItem(name=f.name, portion=f.portion, calories=f.calories, reason=reason))
        return out

    def suggest_activities_kg(self, req: RecommendationRequest) -> list[SuggestedActivity]:
        """Retrieves and ranks wellness tasks dynamically from the Knowledge Graph."""
        out: list[SuggestedActivity] = []
        label = req.affective_state.label if req.affective_state else None

        active_states: list[str] = []
        if label == AffectiveLabel.stress:
            active_states.append("state:stress")
        elif label == AffectiveLabel.anxiety:
            active_states.append("state:anxiety")
        elif label == AffectiveLabel.sad:
            active_states.append("state:sad")

        slots = req.wellness.slots[:3] if req.wellness else []
        if slots:
            avg_fatigue = sum(s.mental_fatigue for s in slots) / len(slots)
            if avg_fatigue > 0.6:
                active_states.append("state:fatigue")
            avg_energy = sum(s.energy for s in slots) / len(slots)
            if avg_energy < 0.4:
                active_states.append("state:low_energy")
            avg_focus = sum(s.focus for s in slots) / len(slots)
            if avg_focus < 0.4:
                active_states.append("state:low_focus")

        is_overload = req.schedule is not None and req.schedule.overload_predicted
        if is_overload:
            active_states.append("state:stress")

        if not active_states:
            return []

        # Retrieve all WELLNESS_TASK nodes from the Knowledge Graph
        task_candidates = []
        for nid, data in self.g.nodes(data=True):
            node = data.get("node")
            if not isinstance(node, KGNode) or node.node_type != NodeType.WELLNESS_TASK:
                continue

            props = node.properties
            duration = int(props.get("duration_minutes", 10))
            category_str = props.get("category", "personal")
            task_category = getattr(TaskCategory, category_str, TaskCategory.personal)

            # Score task via KG edges to active states
            task_score = 0.0
            reasons = []
            for _, tail, edge_data in self.g.out_edges(nid, data=True):
                if tail in active_states:
                    w = float(edge_data.get("weight", 1.0))
                    task_score += w
                    exp = edge_data.get("explanation", "")
                    pmid = edge_data.get("pmid")
                    if pmid:
                        exp = f"{exp} (PMID: {pmid})"
                    if exp and exp not in reasons:
                        reasons.append(exp)

            # GNN representation scoring if task is present in embedding table
            if self.gnn is not None and self.gnn.is_loaded and self.gnn.final_embeddings is not None:
                t_idx = self.gnn.node_to_idx.get(nid)
                if t_idx is not None:
                    for s_id in active_states:
                        s_idx = self.gnn.node_to_idx.get(s_id)
                        if s_idx is not None:
                            sim = float(self.gnn.model.cosine_score(self.gnn.final_embeddings, s_idx, t_idx))
                            task_score += 0.3 * sim

            if is_overload and "planning" in nid:
                task_score += 1.5
            if (label in (AffectiveLabel.stress, AffectiveLabel.anxiety)) and nid == "task:walk_outside":
                task_score += 2.0

            if task_score > 0:
                explanation = "; ".join(reasons) if reasons else props.get("source", "Evidence-based wellness protocol")
                # Specific calibrated descriptions (framed as options for consideration)
                if nid == "task:walk_outside" and (label in (AffectiveLabel.stress, AffectiveLabel.anxiety)):
                    explanation = "Option to consider: A short walk can help bring stress down."
                elif nid == "task:rest_break" and any("fatigue" in s for s in active_states):
                    explanation = "Option to consider: Mental fatigue is forecast to be high; take a short restorative break."
                elif nid == "task:planning_break" and is_overload:
                    explanation = "Option to consider: Your schedule looks overloaded — take a 15-minute planning break."
                elif "heartsteps" in nid:
                    msg = props.get("message", "Take a short walking pause.")
                    empirical = props.get("empirical_stats", {})
                    eval_pos = empirical.get("positive_responses", 0)
                    eval_neg = empirical.get("negative_responses", 0)
                    total_eval = eval_pos + eval_neg
                    sample_size = empirical.get("sample_size", total_eval)
                    unanswered = empirical.get("unanswered_or_snoozed", max(0, sample_size - total_eval))
                    eval_str = (
                        f" (study log: {eval_pos}/{total_eval} positive of rated prompts, {unanswered} unrated of {sample_size} delivered; descriptive trial feedback, not clinical efficacy)"
                        if total_eval > 0
                        else ""
                    )
                    explanation = (
                        f"Option to consider: {msg} "
                        f"[HeartSteps mHealth MRT (Klasnja et al. 2019){eval_str}]"
                    )

                task_candidates.append({
                    "task_id": nid,
                    "title": node.name,
                    "category": task_category,
                    "estimated_minutes": max(5, duration),
                    "reason": explanation,
                    "score": task_score,
                })

        # Rank candidates by task_score
        task_candidates.sort(key=lambda x: x["score"], reverse=True)

        # Select top non-overlapping activities (max 2)
        seen_titles = set()
        for cand in task_candidates:
            if cand["title"] not in seen_titles:
                seen_titles.add(cand["title"])
                out.append(
                    SuggestedActivity(
                        title=cand["title"],
                        category=cand["category"],
                        estimated_minutes=cand["estimated_minutes"],
                        reason=cand["reason"],
                    )
                )
            if len(out) >= 2:
                break

        return out

    def build_rationale(
        self,
        meal: MealType,
        label: AffectiveLabel | None,
        energy: float | None,
        moved: bool,
        is_stale: bool = False,
        original_label: AffectiveLabel | None = None,
        hours_elapsed: float = 0.0,
        is_different_day: bool = False,
        confidence: float | None = None,
        probability: float | None = None,
        profile: DietProfile | None = None,
    ) -> str:
        parts = []
        expiry_hours = self.thresholds.affective_state_expiry_hours

        if is_stale and original_label:
            conf_str = f" (conf: {confidence:.2f})" if confidence is not None else ""
            if is_different_day:
                parts.append(
                    f"affective context ({original_label.value}{conf_str}) from previous day ({hours_elapsed:.1f}h elapsed) has exceeded {expiry_hours:g}h heuristic horizon and is not applied as a current observation; baseline steady-energy foods prioritized"
                )
            else:
                parts.append(
                    f"affective context ({original_label.value}{conf_str}) observed {hours_elapsed:.1f}h ago has exceeded {expiry_hours:g}h heuristic horizon and is not applied as a current observation; baseline steady-energy foods prioritized"
                )
        elif label == AffectiveLabel.stress:
            prob_str = f" ({int(round(probability * 100))}% probability, conf: {confidence:.2f})" if probability is not None else (f" (conf: {confidence:.2f})" if confidence is not None else "")
            parts.append(f"elevated stress detected{prob_str} within recent observation window; steady-energy foods prioritized")
        elif label == AffectiveLabel.anxiety:
            prob_str = f" ({int(round(probability * 100))}% probability, conf: {confidence:.2f})" if probability is not None else (f" (conf: {confidence:.2f})" if confidence is not None else "")
            parts.append(f"elevated anxiety detected{prob_str} within recent observation window; steady-energy foods prioritized")
        elif label == AffectiveLabel.sad:
            parts.append("mood-supporting foods prioritized")

        if profile and profile.health_conditions:
            conds_str = " ".join(profile.health_conditions).lower()
            if "diabetes" in conds_str:
                parts.append("carbohydrate and fiber telemetry aligned with clinical diabetes dietary guidelines (ADA Standards of Care); individual targets should follow endocrinology/dietetic care plans")

        if energy is not None and energy < 0.4:
            parts.append("your energy is forecast to be low around this time")
        if moved:
            parts.append("time adjusted to fit a free gap in your schedule")

        return (
            (f"{meal.value.capitalize()}: " + "; ".join(parts) + ".")
            if parts
            else f"{meal.value.capitalize()}: balanced choice for your profile."
        )

    def recommend(self, req: RecommendationRequest) -> tuple[list[MealRecommendation], list[SuggestedActivity]]:
        tz = ZoneInfo(req.timezone)
        label = req.affective_state.label if req.affective_state else None
        # Auto-sync safeguard: if current label has 0 probability, pick highest probability label
        if req.affective_state and req.affective_state.probabilities:
            probs = req.affective_state.probabilities
            if probs:
                best_label = max(probs, key=probs.get)
                if probs.get(label, 0) == 0 and probs[best_label] > 0:
                    label = best_label

        meals = []
        used: dict[str, int] = {}
        daily = req.profile.daily_calorie_target or DEFAULT_DAILY_KCAL

        # Summary of active dietary constraints for reporting
        constraints = []
        if req.profile.vegan:
            constraints.append("vegan")
        if req.profile.vegetarian:
            constraints.append("vegetarian")
        if req.profile.halal:
            constraints.append("halal")
        if req.profile.allergies:
            constraints.append(f"allergies: {req.profile.allergies}")
        if req.profile.dislikes:
            constraints.append(f"dislikes: {req.profile.dislikes}")
        if req.profile.health_conditions:
            constraints.append(f"conditions: {req.profile.health_conditions}")
        constraints_str = ", ".join(constraints) if constraints else "standard dietary targets"

        for meal, start, end, moved in meal_windows(req.as_of, tz, busy_intervals(req.schedule)):
            meal_local = start.astimezone(tz)
            as_of_local = req.as_of.astimezone(tz)

            # Temporal Freshness: evaluate decay of acute affective states using elapsed time and configurable heuristic
            time_delta = start - req.as_of
            hours_elapsed = max(0.0, time_delta.total_seconds() / 3600.0)
            is_different_day = (meal_local.date() > as_of_local.date())
            is_stale = (time_delta > timedelta(hours=self.thresholds.affective_state_expiry_hours)) or is_different_day

            meal_label = None if is_stale else label
            energy = energy_at(req.wellness, start)
            local_hour = meal_local.hour

            foods = self.pick_foods_kg(meal, meal_label, energy, local_hour, req.profile, used, req.wellness)
            meal_target = daily * CALORIE_SHARE[meal]

            # Extract state probabilities and confidence
            prob_val = None
            conf_val = None
            if req.affective_state:
                conf_val = req.affective_state.confidence
                if req.affective_state.probabilities and label in req.affective_state.probabilities:
                    prob_val = req.affective_state.probabilities[label]

            base_rationale = self.build_rationale(
                meal,
                meal_label,
                energy,
                moved,
                is_stale=is_stale,
                original_label=label,
                hours_elapsed=hours_elapsed,
                is_different_day=is_different_day,
                confidence=conf_val,
                probability=prob_val,
                profile=req.profile,
            )

            if not foods:
                # Explicit reporting when candidate catalog cannot satisfy constraints
                rationale = (
                    f"[STATUS: INCOMPLETE_PLAN] {meal.value.capitalize()}: No candidate foods in current catalog satisfy strict constraints ({constraints_str}). "
                    f"Safe candidate pool exhausted; preserving safety over completeness (fail-closed: no unverified or unsafe foods returned). "
                    f"Next steps: Expand pantry catalog with certified compliant foods or arrange individualized clinical dietary planning."
                )
                meals.append(
                    MealRecommendation(
                        meal_type=meal,
                        window_start=start,
                        window_end=end,
                        foods=[],
                        rationale=rationale,
                    )
                )
                continue

            # Nutritional Completeness Validation across all foods in meal
            meal_cals = sum(f.calories for f in foods)
            selected_objs = [self.foods_by_name.get(f.name) for f in foods if f.name in self.foods_by_name]
            p_tot = sum(getattr(f, 'protein_g', 0.0) or 0.0 for f in selected_objs)
            c_tot = sum(getattr(f, 'carbs_g', 0.0) or 0.0 for f in selected_objs)
            f_tot = sum(getattr(f, 'fat_g', 0.0) or 0.0 for f in selected_objs)

            is_snack = (meal == MealType.snack)
            min_cal_ratio = self.thresholds.min_snack_calorie_ratio if is_snack else self.thresholds.min_meal_calorie_ratio
            min_protein = self.thresholds.min_snack_protein_g if is_snack else self.thresholds.min_main_protein_g

            cal_shortfall = round(meal_target - meal_cals)
            p_shortfall = round(max(0.0, min_protein - p_tot), 1)

            cal_deficit = (meal_cals < min_cal_ratio * meal_target)
            protein_deficit = (p_tot < min_protein)

            if cal_deficit or protein_deficit:
                deficits = []
                if cal_deficit:
                    deficits.append(f"{meal_cals} kcal vs {meal_target:.0f} kcal target (-{cal_shortfall} kcal shortfall)")
                if protein_deficit:
                    deficits.append(f"protein: {p_tot:.1f}g vs {min_protein:.0f}g baseline (-{p_shortfall}g protein shortfall)")
                deficit_summary = "; ".join(deficits)

                rationale = (
                    f"[STATUS: INCOMPLETE_PLAN] {base_rationale} Nutritional deficit alert: selected foods provide {deficit_summary}. "
                    f"Catalog contains limited safe items under strict constraints ({constraints_str}); "
                    f"supplemental calories or individualized planning recommended."
                )
            else:
                rationale = f"[STATUS: COMPLETE] {base_rationale}"

            meals.append(
                MealRecommendation(
                    meal_type=meal,
                    window_start=start,
                    window_end=end,
                    foods=foods,
                    rationale=rationale,
                )
            )
        return meals, self.suggest_activities_kg(req)


_SERVICE_INSTANCE = KnowledgeGraphService()


def recommend_kg(req: RecommendationRequest) -> tuple[list[MealRecommendation], list[SuggestedActivity]]:
    return _SERVICE_INSTANCE.recommend(req)
