"""Runtime configuration, read from environment variables (see the repo's .env.example)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # storage
    database_url: str = "postgresql+psycopg://wellness:wellness_dev_pw@localhost:5432/wellness"
    redis_url: str | None = "redis://localhost:6379/0"

    # auth
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = 60
    refresh_token_days: int = 30

    # module services
    m1_url: str = "http://localhost:8101"
    m2_url: str = "http://localhost:8102"
    m3_url: str = "http://localhost:8103"
    m4_url: str = "http://localhost:8104"
    module_timeout_seconds: float = 10.0

    # pipeline
    pipeline_min_interval_minutes: int = 15
    pipeline_window_minutes: int = 60
    schedule_horizon_days: int = 7

    # http
    cors_origins: str = "*"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
