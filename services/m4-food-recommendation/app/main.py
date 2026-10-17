"""M4 Context-Aware Food Recommendation service (port 8104)."""

from fastapi import FastAPI

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
    def recommendations(req: RecommendationRequest) -> Recommendations:
        return pred.predict(req)

    return app


app = create_app()
