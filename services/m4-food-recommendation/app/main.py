"""M4 Context-Aware Food Recommendation service (port 8104)."""

from fastapi import Body, FastAPI

from wellness_contracts import RecommendationRequest, Recommendations

from .predictor import Predictor
from .settings import settings


def create_app(predictor: Predictor | None = None) -> FastAPI:
    pred = predictor or Predictor()
    app = FastAPI(
        title="M4 — Context-Aware Food Recommendation",
        version=pred.info.version,
        description="Owner: T.C. Gunaweera. Affective state + wellness + schedule + diet profile → meals.",
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "service": settings.service_name, "model": pred.info.model_dump()}

    @app.post("/v1/recommendations", response_model=Recommendations)
    def recommendations(
        req: RecommendationRequest = Body(
            openapi_examples={
                "stressed_context": {
                    "summary": "Stressed user with fatigue (Ready to run)",
                    "value": {
                        "user_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                        "as_of": "2026-10-21T04:00:00Z",
                        "timezone": "Asia/Colombo",
                        "affective_state": {
                            "user_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                            "window_start": "2026-10-21T03:00:00Z",
                            "window_end": "2026-10-21T04:00:00Z",
                            "label": "stress",
                            "probabilities": {
                                "stress": 0.85,
                                "anxiety": 0.1,
                                "relaxed": 0.05,
                                "happy": 0.0,
                                "sad": 0.0,
                            },
                            "confidence": 0.85,
                            "features": {},
                            "model": {"name": "m1-affective-state", "version": "0.1.0"},
                        },
                        "wellness": {
                            "user_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                            "generated_at": "2026-10-21T04:00:00Z",
                            "slot_minutes": 60,
                            "slots": [
                                {
                                    "start": "2026-10-21T04:00:00Z",
                                    "end": "2026-10-21T05:00:00Z",
                                    "mental_fatigue": 0.8,
                                    "focus": 0.4,
                                    "energy": 0.3,
                                    "burnout_risk": 0.7,
                                }
                            ],
                            "sleep_hours_last_night": 6.5,
                            "model": {"name": "m2-wellness", "version": "0.1.0"},
                        },
                        "schedule": {
                            "user_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                            "generated_at": "2026-10-21T04:00:00Z",
                            "horizon_start": "2026-10-21T04:00:00Z",
                            "horizon_end": "2026-10-22T04:00:00Z",
                            "items": [],
                            "unscheduled": [],
                            "at_risk_task_ids": [],
                            "overload_predicted": False,
                            "reason": "scheduled",
                            "model": {"name": "m3-schedule", "version": "0.1.0"},
                        },
                        "profile": {
                            "vegetarian": False,
                            "vegan": False,
                            "halal": False,
                            "allergies": [],
                            "dislikes": [],
                            "health_conditions": [],
                        },
                    },
                }
            }
        ),
    ) -> Recommendations:
        return pred.predict(req)

    return app


app = create_app()
