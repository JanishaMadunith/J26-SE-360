"""Module 4 predictor.

Knowledge Graph Context-Aware Recommender (v0.2.0):
Traverses the domain knowledge graph connecting user affective/cognitive states
to bio-nutrients and food items, with strict safety constraint pruning.
"""

from m4_food.graph.kg_service import recommend_kg
from wellness_contracts import ModelInfo, RecommendationRequest, Recommendations, utc_now


class Predictor:
    info = ModelInfo(name="m4-kg-gnn-recommender", version="0.3.0")

    def predict(self, req: RecommendationRequest) -> Recommendations:
        meals, activities = recommend_kg(req)
        return Recommendations(
            user_id=req.user_id,
            generated_at=utc_now(),
            meals=meals,
            suggested_activities=activities,
            model=self.info,
        )
