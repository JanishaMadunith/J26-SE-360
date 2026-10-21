"""Knowledge Graph Service Adapter for the M4 FastAPI Endpoint.

Translates wellness_contracts.RecommendationRequest into Knowledge Graph traversal,
safety filtering, context scoring, and returns MealRecommendation and SuggestedActivity contracts.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

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
from .schema import RelationType


class KnowledgeGraphService:
    def __init__(self, foods_list: list[Food] | None = None, enable_gnn: bool = True):
        self.foods = foods_list if foods_list is not None else FOODS
        self.g = create_default_knowledge_graph(self.foods)
        self.foods_by_name = {f.name: f for f in self.foods}
        self.gnn = GNNRecommenderRanker() if enable_gnn else None

    def _fid(self, name: str) -> str:
        return f"food:{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}"

    def is_safe_kg(self, food: Food, profile: DietProfile) -> bool:
        """Graph-based safety and dietary compliance check."""
        fid = self._fid(food.name)
        if not self.g.has_node(fid):
            return False

        # 1. Diet compliance
        if profile.vegan and not self.g.has_edge(fid, "diet:vegan"):
            return False
        if profile.vegetarian and not self.g.has_edge(fid, "diet:vegetarian"):
            return False
        if profile.halal and not self.g.has_edge(fid, "diet:halal"):
            return False

        # 2. Allergens check (both graph edges and direct name match)
        food_name_lower = food.name.lower()
        for a in (x.lower().strip() for x in profile.allergies):
            if not a:
                continue
            if a in food.tags or a in food_name_lower:
                return False
            # Check allergen edges in KG
            for _, tail, data in self.g.out_edges(fid, data=True):
                if data.get("relation") == RelationType.HAS_ALLERGEN:
                    if a in tail.lower():
                        return False

        # 3. Dislikes
        for d in (x.lower().strip() for x in profile.dislikes):
            if d and d in food_name_lower:
                return False

        # 4. Medical / Health Condition Contraindications
        conditions_str = " ".join(profile.health_conditions).lower()
        if "diabet" in conditions_str:
            # Check if food has nutrient with CONTRAINDICATED_FOR condition:diabetes
            if self.g.has_edge(fid, "nutrient:high-GI"):
                return False
        if "hypertension" in conditions_str or "blood pressure" in conditions_str:
            if self.g.has_edge(fid, "nutrient:high-sodium"):
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
    ) -> tuple[float, list[str]]:
        """Scores candidate food using Knowledge Graph path traversal."""
        fid = self._fid(food.name)
        tags = food.tags
        score = 0.0
        reasons = []

        # Knowledge Graph Path Traversal:
        # Determine active states for the meal context
        active_states = []
        if label in (AffectiveLabel.stress, AffectiveLabel.anxiety):
            active_states.append("state:stress")
            active_states.append("state:anxiety")
        elif label == AffectiveLabel.sad:
            active_states.append("state:sad")

        if energy is not None and energy < 0.4:
            active_states.append("state:low_energy")

        # Traverse: Food -> Nutrient -> Active States
        for _, nutrient_id, edge_data in self.g.out_edges(fid, data=True):
            if edge_data.get("relation") != RelationType.CONTAINS:
                continue

            for state_id in active_states:
                if self.g.has_edge(nutrient_id, state_id):
                    st_data = self.g.get_edge_data(nutrient_id, state_id)
                    w = float(st_data.get("weight", 1.0))
                    score += w
                    exp = st_data.get("explanation", "")
                    if exp and exp not in reasons:
                        reasons.append(exp)

        # Negative factors from nutrients
        if label in (AffectiveLabel.stress, AffectiveLabel.anxiety):
            if "caffeine" in tags:
                score -= 1.0
            if "fried" in tags:
                score -= 0.6
            if "high-sugar" in tags:
                score -= 0.5

        if energy is not None and energy < 0.4:
            if "high-sugar" in tags:
                score -= 0.5

        if "low-GI" in tags:
            score += 0.3
            if "low glycaemic index avoids energy crashes" not in reasons and "diabet" in reasons:
                reasons.append("low glycaemic index avoids energy crashes")

        if "high-sodium" in tags:
            score -= 0.4
        if "fried" in tags:
            score -= 0.3
        if "high-GI" in tags:
            score -= 0.2

        # Caffeine restriction after 15:00
        if "caffeine" in tags and local_hour >= 15:
            score -= 2.0
            reasons.append("caffeine avoided after 15:00 to protect sleep")

        # GNN Neural Representation Score
        gnn_score_val = 0.0
        if self.gnn is not None and self.gnn.is_loaded and self.gnn.final_embeddings is not None:
            f_idx = self.gnn.node_to_idx.get(fid)
            if f_idx is not None:
                for state_id in active_states:
                    s_idx = self.gnn.node_to_idx.get(state_id)
                    if s_idx is not None:
                        sim = self.gnn.model.score(self.gnn.final_embeddings, s_idx, f_idx)
                        gnn_score_val += sim
                        score += 0.3 * sim

        if gnn_score_val != 0.0:
            reasons.insert(0, f"[GNN Score: {gnn_score_val:+.2f}]")

        # Calorie proximity
        score -= 0.6 * abs(food.calories - kcal_target) / max(kcal_target, 1)

        return score, reasons

    def pick_foods_kg(
        self,
        meal: MealType,
        label: AffectiveLabel | None,
        energy: float | None,
        local_hour: int,
        profile: DietProfile,
        used: dict[str, int],
    ) -> list[FoodItem]:
        daily = profile.daily_calorie_target or DEFAULT_DAILY_KCAL
        n = ITEMS_PER_MEAL[meal]
        target = daily * CALORIE_SHARE[meal] / n

        # Filter safe foods using Knowledge Graph
        cands = [f for f in self.foods if meal in f.meals and self.is_safe_kg(f, profile)]
        scored = []
        for f in cands:
            sc, why = self.score_food_kg(f, meal, label, energy, local_hour, target)
            penalty = VARIETY_PENALTY * used.get(f.name, 0)
            scored.append(((sc - penalty, why), f))

        ranked = sorted(scored, key=lambda x: (-x[0][0], x[1].name))
        out = []
        for (_, why), f in ranked[:n]:
            used[f.name] = used.get(f.name, 0) + 1
            reason = "; ".join(why) if why else "fits your profile and the time of day"
            out.append(FoodItem(name=f.name, portion=f.portion, calories=f.calories, reason=reason))
        return out

    def suggest_activities_kg(self, req: RecommendationRequest) -> list[SuggestedActivity]:
        out: list[SuggestedActivity] = []
        label = req.affective_state.label if req.affective_state else None

        if label in (AffectiveLabel.stress, AffectiveLabel.anxiety):
            out.append(
                SuggestedActivity(
                    title="10-minute walk outside",
                    category=TaskCategory.exercise,
                    estimated_minutes=10,
                    reason="A short walk can help bring stress down.",
                )
            )

        slots = req.wellness.slots[:3] if req.wellness else []
        if slots and sum(s.mental_fatigue for s in slots) / len(slots) > 0.7:
            out.append(
                SuggestedActivity(
                    title="20-minute rest break",
                    category=TaskCategory.personal,
                    estimated_minutes=20,
                    reason="Mental fatigue is forecast to be high.",
                )
            )

        if req.schedule is not None and req.schedule.overload_predicted and len(out) < 2:
            out.append(
                SuggestedActivity(
                    title="15-minute planning break",
                    category=TaskCategory.personal,
                    estimated_minutes=15,
                    reason="Your schedule looks overloaded — re-prioritise.",
                )
            )

        return out[:2]

    def build_rationale(self, meal: MealType, label: AffectiveLabel | None, energy: float | None, moved: bool) -> str:
        parts = []
        if label in (AffectiveLabel.stress, AffectiveLabel.anxiety):
            parts.append("you seem stressed, so calming, steady-energy foods are preferred")
        elif label == AffectiveLabel.sad:
            parts.append("mood-supporting foods are preferred")
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
        meals = []
        used: dict[str, int] = {}

        for meal, start, end, moved in meal_windows(req.as_of, tz, busy_intervals(req.schedule)):
            energy = energy_at(req.wellness, start)
            local_hour = start.astimezone(tz).hour
            foods = self.pick_foods_kg(meal, label, energy, local_hour, req.profile, used)
            if not foods:
                continue
            meals.append(
                MealRecommendation(
                    meal_type=meal,
                    window_start=start,
                    window_end=end,
                    foods=foods,
                    rationale=self.build_rationale(meal, label, energy, moved),
                )
            )
        return meals, self.suggest_activities_kg(req)


_SERVICE_INSTANCE = KnowledgeGraphService()


def recommend_kg(req: RecommendationRequest) -> tuple[list[MealRecommendation], list[SuggestedActivity]]:
    return _SERVICE_INSTANCE.recommend(req)
