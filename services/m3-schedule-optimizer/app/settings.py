"""Service settings — override with environment variables prefixed M3_ (e.g. M3_LOG_LEVEL=debug)."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="M3_", extra="ignore")

    service_name: str = "m3-schedule-optimizer"
    port: int = 8103
    log_level: str = "info"
    models_dir: str = "models"


settings = Settings()
