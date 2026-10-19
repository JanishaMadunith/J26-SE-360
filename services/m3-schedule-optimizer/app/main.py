"""M3 Predictive Adaptive Schedule Optimization service (port 8103).

GET  /health               service + model status
POST /v1/schedule          Module 3 contract: tasks + wellness forecast -> schedule with completion chances
...  /v1/completion, /v1/tasktime, /v1/demo   model endpoints (see app/demo.py)
GET  /demo                 demo page for the progress presentation
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from wellness_contracts import ScheduleRequest, ScheduleResult

from .demo import router as demo_router
from .predictor import Predictor
from .settings import settings

STATIC = Path(__file__).parent / "static"


def create_app(predictor: Predictor | None = None) -> FastAPI:
    pred = predictor or Predictor()
    app = FastAPI(
        title="M3 — Predictive Adaptive Schedule Optimization",
        version=pred.info.version,
        description="Owner: G.W.D.D.N. Kumarasinghe. Tasks + wellness forecast → optimised schedule "
        "with completion probabilities. Demo page: /demo",
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "service": settings.service_name, "model": pred.status()}

    @app.post("/v1/schedule", response_model=ScheduleResult)
    def make_schedule(req: ScheduleRequest) -> ScheduleResult:
        return pred.predict(req)

    @app.get("/demo", response_class=HTMLResponse, include_in_schema=False)
    def demo() -> str:
        return (STATIC / "demo.html").read_text(encoding="utf-8")

    app.include_router(demo_router(pred))
    return app


app = create_app()
