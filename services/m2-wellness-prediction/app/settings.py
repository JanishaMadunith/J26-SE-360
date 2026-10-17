"""Service settings — override with environment variables prefixed M2_ (e.g. M2_LOG_LEVEL=debug)."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="M2_", extra="ignore")

    service_name: str = "m2-wellness-prediction"
    port: int = 8102
    log_level: str = "info"
    models_dir: str = "models"


settings = Settings()
