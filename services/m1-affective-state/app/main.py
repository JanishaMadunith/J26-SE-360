"""M1 Affective State Detection service (port 8101)."""

from fastapi import FastAPI

from wellness_contracts import AffectiveStateRequest, AffectiveStateResult

from .predictor import Predictor
from .settings import settings


def create_app(predictor: Predictor | None = None) -> FastAPI:
    pred = predictor or Predictor()
    app = FastAPI(
        title="M1 — Affective State Detection",
        version=pred.info.version,
        description="Owner: J.D. Jayatilake. Raw phone signals → current affective state.",
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "service": settings.service_name, "model": pred.info.model_dump()}

    @app.post("/v1/affective-state", response_model=AffectiveStateResult)
    def affective_state(req: AffectiveStateRequest) -> AffectiveStateResult:
        return pred.predict(req)

    return app


app = create_app()
