"""M2 Cognitive Wellness Prediction service (port 8102)."""

from fastapi import FastAPI

from wellness_contracts import WellnessForecast, WellnessForecastRequest

from .predictor import Predictor
from .settings import settings


def create_app(predictor: Predictor | None = None) -> FastAPI:
    pred = predictor or Predictor()
    app = FastAPI(
        title="M2 — Cognitive Wellness Prediction",
        version=pred.info.version,
        description="Owner: J.M. Samarathunga. Affective history + behaviour → slot-level wellness forecast.",
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "service": settings.service_name, "model": pred.info.model_dump()}

    @app.post("/v1/wellness-forecast", response_model=WellnessForecast)
    def wellness_forecast(req: WellnessForecastRequest) -> WellnessForecast:
        return pred.predict(req)

    return app


app = create_app()
