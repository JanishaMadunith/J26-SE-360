"""HTTP clients for the four module services. Every call validates both directions against the
shared contracts, so a module returning a malformed payload fails loudly instead of corrupting data."""

import httpx
from pydantic import BaseModel, ValidationError

from wellness_contracts import (
    AffectiveStateRequest,
    AffectiveStateResult,
    RecommendationRequest,
    Recommendations,
    ScheduleRequest,
    ScheduleResult,
    WellnessForecast,
    WellnessForecastRequest,
)

from ..config import Settings


class ModuleError(Exception):
    """A module call failed (unreachable, timeout, HTTP error or invalid response)."""

    def __init__(self, module: str, message: str):
        super().__init__(f"{module}: {message}")
        self.module = module


class ModuleClient:
    def __init__(self, name: str, base_url: str, path: str, response_model: type[BaseModel], client: httpx.Client):
        self.name, self.base_url, self.path = name, base_url.rstrip("/"), path
        self.response_model = response_model
        self._client = client

    def call(self, payload: BaseModel):
        url = f"{self.base_url}{self.path}"
        try:
            r = self._client.post(url, content=payload.model_dump_json(), headers={"Content-Type": "application/json"})
        except httpx.TimeoutException as e:
            raise ModuleError(self.name, f"timeout calling {url}") from e
        except httpx.HTTPError as e:
            raise ModuleError(self.name, f"cannot reach {url}: {e.__class__.__name__}") from e
        if r.status_code != 200:
            raise ModuleError(self.name, f"HTTP {r.status_code}: {r.text[:300]}")
        try:
            return self.response_model.model_validate_json(r.content)
        except ValidationError as e:
            raise ModuleError(self.name, f"response does not match contract: {e.errors()[:3]}") from e

    def healthy(self, timeout: float = 1.5) -> bool:
        try:
            r = self._client.get(f"{self.base_url}/health", timeout=timeout)
            return r.status_code == 200
        except httpx.HTTPError:
            return False


class ModuleClients:
    """Bundle of the four clients sharing one connection pool. Pass `transport` in tests to fake modules."""

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None):
        self._http = httpx.Client(timeout=settings.module_timeout_seconds, transport=transport)
        self.m1 = ModuleClient("m1", settings.m1_url, "/v1/affective-state", AffectiveStateResult, self._http)
        self.m2 = ModuleClient("m2", settings.m2_url, "/v1/wellness-forecast", WellnessForecast, self._http)
        self.m3 = ModuleClient("m3", settings.m3_url, "/v1/schedule", ScheduleResult, self._http)
        self.m4 = ModuleClient("m4", settings.m4_url, "/v1/recommendations", Recommendations, self._http)

    def affective_state(self, req: AffectiveStateRequest) -> AffectiveStateResult:
        return self.m1.call(req)

    def wellness_forecast(self, req: WellnessForecastRequest) -> WellnessForecast:
        return self.m2.call(req)

    def schedule(self, req: ScheduleRequest) -> ScheduleResult:
        return self.m3.call(req)

    def recommendations(self, req: RecommendationRequest) -> Recommendations:
        return self.m4.call(req)

    def health(self) -> dict[str, bool]:
        return {c.name: c.healthy() for c in (self.m1, self.m2, self.m3, self.m4)}

    def close(self) -> None:
        self._http.close()
