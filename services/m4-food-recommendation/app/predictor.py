"""Module 4 predictor.

BASELINE (v0.1.0): rule-based context-aware recommender (src/m4_food/recommender.py) over a small
built-in food table (src/m4_food/foods.py).

REPLACE with the trained recommender (e.g. learning-to-rank over a full nutrition database with user
feedback): keep the class name, `info` and `predict()` signature.
"""

from m4_food.recommender import recommend
from wellness_contracts import ModelInfo, RecommendationRequest, Recommendations, utc_now


class Predictor:
    info = ModelInfo(name="m4-rule-baseline", version="0.1.0")

    def predict(self, req: RecommendationRequest) -> Recommendations:
        meals, activities = recommend(req)
        return Recommendations(
            user_id=req.user_id, generated_at=utc_now(), meals=meals, suggested_activities=activities, model=self.info
        )
