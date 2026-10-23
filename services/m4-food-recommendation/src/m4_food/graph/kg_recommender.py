"""Knowledge Graph Recommender: Graph Traversal, Safety Filtering, Path-Based Scoring, and Explanation Generator."""

from __future__ import annotations

import networkx as nx
try:
    from .gnn_ranker import GNNRecommenderRanker
    from .kg_service import KnowledgeGraphService
    from .knowledge_base import create_default_knowledge_graph
    from .schema import NodeType, RawUserContext, RelationType
except ImportError:
    from gnn_ranker import GNNRecommenderRanker
    from kg_service import KnowledgeGraphService
    from knowledge_base import create_default_knowledge_graph
    from schema import NodeType, RawUserContext, RelationType
from wellness_contracts import DietProfile


class KnowledgeGraphRecommender:
    """Core recommendation engine operating on the domain Knowledge Graph and GNN."""

    def __init__(self, graph: nx.DiGraph | None = None, enable_gnn: bool = True, use_heuristic_weights: bool = False):
        self.use_heuristic_weights = use_heuristic_weights
        self.g = graph if graph is not None else create_default_knowledge_graph(use_heuristic_weights=use_heuristic_weights)
        self.gnn = GNNRecommenderRanker() if enable_gnn else None
        self.service = KnowledgeGraphService(foods_list=None, enable_gnn=enable_gnn)

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
        """Applies unified deterministic safety constraints using canonical KnowledgeGraphService logic."""
        node_data = self.g.nodes[food_id].get("node")
        food_name = node_data.name if node_data else food_id.replace("food:", "").replace("_", " ")
        food = self.service.foods_by_name.get(food_name)
        if not food:
            return False, "Food metadata not found in catalog"

        prof = DietProfile(
            vegetarian=ctx.is_vegetarian,
            vegan=ctx.is_vegan,
            halal=ctx.is_halal,
            allergies=list(ctx.allergies),
            dislikes=list(ctx.dislikes),
            health_conditions=list(ctx.health_conditions),
            daily_calorie_target=ctx.daily_calorie_target,
        )
        is_safe = self.service.is_safe_kg(food, prof)
        if not is_safe:
            return False, "Failed safety / dietary constraint filter"
        return True, "Safe"

    def recommend(
        self, ctx: RawUserContext, top_k_foods: int = 3, top_k_tasks: int = 2, mode: str = "model_c"
    ) -> dict:
        """Generates context-aware, KG-reasoned food and task recommendations from raw user input.
        
        Ablation Modes:
            - 'model_a': Content-based baseline without dynamic affective/cognitive context.
            - 'model_b': Pure LightGCN embedding ranking without symbolic KG paths.
            - 'model_c': Neuro-symbolic hybrid (LightGCN + KG with EQUAL 1.0 weights baseline) [Default].
            - 'model_d': Neuro-symbolic hybrid (LightGCN + KG with evidence-tier weights 0.70-0.95).
        """
        active_states = self._extract_active_states(ctx) if mode != "model_a" else []

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

            # Deterministic Early Safety Pruning (Allergen & Contraindication CSP)
            safe, reason = self._is_food_safe(nid, ctx)
            if not safe:
                continue

            # KG Path Traversal: Food -> Nutrient -> Active User State
            path_score = 0.0
            reasons = []

            if mode in ("model_c", "model_d"):
                for _, nutrient_id, n_edge in self.g.out_edges(nid, data=True):
                    if n_edge.get("relation") != RelationType.CONTAINS:
                        continue

                    nutrient_node = self.g.nodes[nutrient_id]["node"]

                    for state_id in active_states:
                        if self.g.has_edge(nutrient_id, state_id):
                            s_edge = self.g.get_edge_data(nutrient_id, state_id)
                            # Mode D uses heuristic_tier_weight if present, Mode C uses baseline 1.0
                            if mode == "model_d" and "heuristic_tier_weight" in s_edge:
                                s_w = float(s_edge.get("heuristic_tier_weight", 1.0))
                            else:
                                s_w = float(s_edge.get("weight", 1.0))
                            w = float(n_edge.get("weight", 1.0)) * s_w
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
            if mode != "model_a" and self.gnn is not None and self.gnn.is_loaded and self.gnn.final_embeddings is not None:
                f_idx = self.gnn.node_to_idx.get(nid)
                if f_idx is not None:
                    for state_id in active_states:
                        s_idx = self.gnn.node_to_idx.get(state_id)
                        if s_idx is not None:
                            gnn_score += self.gnn.model.score(self.gnn.final_embeddings, s_idx, f_idx)

            if mode == "model_a":
                # Content-based only: baseline meal suitability + calorie match
                final_food_score = 1.0 - time_fit_penalty - cal_penalty
            elif mode == "model_b":
                # LightGCN only: pure embedding dot product
                final_food_score = gnn_score - time_fit_penalty - cal_penalty
            else:
                # Mode C and Mode D: Hybrid fusion
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
