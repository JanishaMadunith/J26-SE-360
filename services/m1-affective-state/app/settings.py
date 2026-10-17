"""Service settings — override with environment variables prefixed M1_ (e.g. M1_LOG_LEVEL=debug)."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="M1_", extra="ignore")

    service_name: str = "m1-affective-state"
    port: int = 8101
    log_level: str = "info"
    models_dir: str = "models"


settings = Settings()
