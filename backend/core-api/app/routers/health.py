from fastapi import APIRouter, Depends, Request
from sqlalchemy import text

from ..deps import get_module_clients
from ..schemas import HealthOut
from ..services.module_clients import ModuleClients

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut)
def health(request: Request, clients: ModuleClients = Depends(get_module_clients)):
    """'ok' when the database and all four module services respond; otherwise 'degraded'."""
    try:
        with request.app.state.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception:  # noqa: BLE001
        db_ok = False
    modules = clients.health()
    return HealthOut(status="ok" if db_ok and all(modules.values()) else "degraded", db=db_ok, modules=modules)
