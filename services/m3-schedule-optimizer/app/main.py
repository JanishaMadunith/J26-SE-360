"""M3 Predictive Adaptive Schedule Optimization service (port 8103)."""

from fastapi import FastAPI

from wellness_contracts import ScheduleRequest, ScheduleResult

from .predictor import Predictor
from .settings import settings


def create_app(predictor: Predictor | None = None) -> FastAPI:
    pred = predictor or Predictor()
    app = FastAPI(
        title="M3 — Predictive Adaptive Schedule Optimization",
        version=pred.info.version,
        description="Owner: G.W.D.D.N. Kumarasinghe. Tasks + wellness forecast → optimised schedule "
        "with completion probabilities.",
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "service": settings.service_name, "model": pred.info.model_dump()}

    @app.post("/v1/schedule", response_model=ScheduleResult)
    def make_schedule(req: ScheduleRequest) -> ScheduleResult:
        return pred.predict(req)

    return app


app = create_app()
