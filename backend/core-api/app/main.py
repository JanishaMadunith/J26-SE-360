"""J26-SE-360 core API — entry point: `uvicorn app.main:app`."""

import logging

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from .config import Settings, get_settings
from .db import build_engine
from .routers import auth, health, insights, me, pipeline, recommendations, schedule, sensing, tasks

API_DESCRIPTION = """
Core API for the AI-Driven Cognitive Wellness Framework (SLIIT IT4010, J26-SE-360).

Handles accounts, tasks and raw sensor uploads from the mobile app, and orchestrates the four module
services: **M1** affective state → **M2** wellness forecast → **M3** schedule → **M4** recommendations.
All request/response payloads shared with the app or modules are defined in `wellness_contracts`.
"""


def create_app(
    settings: Settings | None = None, engine: Engine | None = None, module_transport: httpx.BaseTransport | None = None
) -> FastAPI:
    settings = settings or get_settings()
    engine = engine or build_engine(settings.database_url)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if settings.jwt_secret == Settings.model_fields["jwt_secret"].default or len(settings.jwt_secret) < 32:
        logging.getLogger("core").warning(
            "JWT_SECRET is the default or shorter than 32 characters — "
            "set a long random value outside local development"
        )

    app = FastAPI(title="J26-SE-360 Core API", version="0.1.0", description=API_DESCRIPTION)
    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    app.state.module_transport = module_transport  # tests inject httpx.MockTransport here

    origins = settings.cors_origin_list
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=origins != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    for r in (health, auth, me, sensing, tasks, schedule, insights, recommendations, pipeline):
        app.include_router(r.router)
    return app


app = create_app()
