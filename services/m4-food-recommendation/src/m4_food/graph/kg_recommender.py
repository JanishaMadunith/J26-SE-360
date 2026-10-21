"""Knowledge Graph Recommender: Graph Traversal, Safety Filtering, Path-Based Scoring, and Explanation Generator."""

from __future__ import annotations

import networkx as nx
try:
    from .gnn_ranker import GNNRecommenderRanker
    from .knowledge_base import create_default_knowledge_graph
    from .schema import NodeType, RawUserContext, RelationType
except ImportError:
    from gnn_ranker import GNNRecommenderRanker
    from knowledge_base import create_default_knowledge_graph
    from schema import NodeType, RawUserContext, RelationType


class KnowledgeGraphRecommender:
    """Core recommendation engine operating on the domain Knowledge Graph and GNN."""

    def __init__(self, graph: nx.DiGraph | None = None, enable_gnn: bool = True):
        self.g = graph if graph is not None else create_default_knowledge_graph()
        self.gnn = GNNRecommenderRanker() if enable_gnn else None

    def _extract_active_states(self, ctx: RawUserContext) -> list[str]:
        """Maps raw user signals (affective + cognitive) to KG state node IDs."""
        active = []
        aff = (ctx.affective_state or "").lower()
        if "stress" in aff:
            active.append("state:stress")
        if "anx" in aff:
            active.append("state:anxiety")
        if "sad" in aff:
            active.append("state:sad")

        if ctx.mental_fatigue >= 0.5:
            active.append("state:fatigue")
        if ctx.energy_level < 0.5:
            active.append("state:low_energy")
        if ctx.focus_level < 0.5:
            active.append("state:low_focus")

        # If perfectly neutral/balanced, add gentle wellness maintenance
        if not active:
            active.append("state:low_focus")
        return list(dict.fromkeys(active))

    def _is_food_safe(self, food_id: str, ctx: RawUserContext) -> tuple[bool, str]:
        """Applies hard constraints (allergens, dietary restrictions, medical conditions)."""
        # 1. Allergens Check
        for _, tail, data in self.g.out_edges(food_id, data=True):
            if data.get("relation") == RelationType.HAS_ALLERGEN:
                allergen_name = tail.replace("allergen:", "").lower()
                for user_all in ctx.allergies:
                    if user_all.lower() in allergen_name or allergen_name in user_all.lower():
                        return False, f"Contains allergen: {allergen_name}"

        # 2. Dietary Preferences (Vegan / Vegetarian / Halal)
        if ctx.is_vegan:
            if not self.g.has_edge(food_id, "diet:vegan"):
                return False, "Not certified vegan"
        if ctx.is_vegetarian:
            if not self.g.has_edge(food_id, "diet:vegetarian"):
                return False, "Not vegetarian"
        if ctx.is_halal:
            if not self.g.has_edge(food_id, "diet:halal"):
                return False, "Not halal"

        # 3. Medical / Health Condition Contraindications
        for cond in ctx.health_conditions:
            cond_node = f"condition:{cond.lower()}"
            if self.g.has_node(cond_node):
                # Check if any nutrient contained in this food is contraindicated for this condition
                for _, nutrient_id, n_data in self.g.out_edges(food_id, data=True):
                    if n_data.get("relation") == RelationType.CONTAINS:
                        if self.g.has_edge(nutrient_id, cond_node):
                            edge_info = self.g.get_edge_data(nutrient_id, cond_node)
                            return False, f"Contraindicated for {cond}: {edge_info.get('explanation')}"

        return True, "Safe"

    def recommend(
        self, ctx: RawUserContext, top_k_foods: int = 3, top_k_tasks: int = 2
    ) -> dict:
        """Generates context-aware, KG-reasoned food and task recommendations from raw user input."""
        active_states = self._extract_active_states(ctx)

        # ---------------------------------------------------------------------
        # 1. FOOD RECOMMENDATIONS VIA KG PATH TRAVERSAL
        # ---------------------------------------------------------------------
        food_candidates = []
        for nid, data in self.g.nodes(data=True):
            node_obj = data.get("node")
            if not node_obj or node_obj.node_type != NodeType.FOOD:
                continue

            # Check meal slot compatibility
            props = node_obj.properties
            if ctx.meal_type and ctx.meal_type.lower() not in [m.lower() for m in props.get("meals", [])]:
                continue

            # Hard safety pruning
            safe, reason = self._is_food_safe(nid, ctx)
            if not safe:
                continue

            # KG Path Traversal: Food -> Nutrient -> Active User State
            path_score = 0.0
            reasons = []

            for _, nutrient_id, n_edge in self.g.out_edges(nid, data=True):
                if n_edge.get("relation") != RelationType.CONTAINS:
                    continue

                nutrient_node = self.g.nodes[nutrient_id]["node"]

                for state_id in active_states:
                    if self.g.has_edge(nutrient_id, state_id):
                        s_edge = self.g.get_edge_data(nutrient_id, state_id)
                        w = float(n_edge.get("weight", 1.0)) * float(s_edge.get("weight", 1.0))
                        path_score += w
                        reasons.append(
                            f"Contains {nutrient_node.name} which {s_edge.get('explanation', 'supports this state')}."
                        )

            # Preparation Time Adaptation
            prep_min = props.get("prep_minutes", 15)
            time_fit_penalty = 0.0
            if prep_min > ctx.available_free_minutes:
                # Excess prep time penalizes score
                time_fit_penalty = (prep_min - ctx.available_free_minutes) * 0.15

            # Calorie Proximity (Soft Constraint)
            cal_diff = abs(props.get("calories", 500) - ctx.target_meal_calories)
            cal_penalty = min(cal_diff / 500.0, 1.0) * 0.2

            # GNN Learned Representation Score
            gnn_score = 0.0
            if self.gnn is not None and self.gnn.is_loaded and self.gnn.final_embeddings is not None:
                f_idx = self.gnn.node_to_idx.get(nid)
                if f_idx is not None:
                    for state_id in active_states:
                        s_idx = self.gnn.node_to_idx.get(state_id)
                        if s_idx is not None:
                            gnn_score += self.gnn.model.score(self.gnn.final_embeddings, s_idx, f_idx)

            final_food_score = path_score + 0.4 * gnn_score - time_fit_penalty - cal_penalty

            # Combine explanations (deduplicated)
            unique_reasons = list(dict.fromkeys(reasons))
            bio_note = props.get("bio_rationale", "")
            explanation_text = " ".join(unique_reasons[:2])
            if bio_note:
                explanation_text = f"{explanation_text} {bio_note}".strip()

            food_candidates.append({
                "food_id": nid,
                "name": node_obj.name,
                "calories": props.get("calories"),
                "prep_minutes": prep_min,
                "score": round(final_food_score, 3),
                "relevance_paths": len(reasons),
                "rationale": explanation_text or "Meets your nutritional and time requirements.",
            })

        # Sort food candidates by score descending
        ranked_foods = sorted(food_candidates, key=lambda x: x["score"], reverse=True)[:top_k_foods]

        # ---------------------------------------------------------------------
        # 2. WELLNESS TASK RECOMMENDATIONS VIA KG PATH TRAVERSAL
        # ---------------------------------------------------------------------
        task_candidates = []
        for nid, data in self.g.nodes(data=True):
            node_obj = data.get("node")
            if not node_obj or node_obj.node_type != NodeType.WELLNESS_TASK:
                continue

            props = node_obj.properties
            duration = props.get("duration_minutes", 5)

            # Filter out tasks that exceed available free time
            if duration > ctx.available_free_minutes:
                continue

            task_score = 0.0
            task_reasons = []

            for state_id in active_states:
                if self.g.has_edge(nid, state_id):
                    edge_data = self.g.get_edge_data(nid, state_id)
                    w = float(edge_data.get("weight", 1.0))
                    task_score += w
                    task_reasons.append(edge_data.get("explanation", ""))

            # Shorter tasks get slight priority when time is very tight
            efficiency_bonus = 0.2 if duration <= ctx.available_free_minutes else 0.0
            total_task_score = task_score + efficiency_bonus

            task_candidates.append({
                "task_id": nid,
                "title": node_obj.name,
                "category": props.get("category", "wellness"),
                "duration_minutes": duration,
                "score": round(total_task_score, 3),
                "rationale": " ".join(list(dict.fromkeys(task_reasons))) or props.get("rationale", ""),
            })

        ranked_tasks = sorted(task_candidates, key=lambda x: x["score"], reverse=True)[:top_k_tasks]

        return {
            "active_context_states": active_states,
            "recommended_foods": ranked_foods,
            "recommended_tasks": ranked_tasks,
        }
