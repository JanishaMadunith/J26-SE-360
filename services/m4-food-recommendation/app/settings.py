"""Service settings — override with environment variables prefixed M4_ (e.g. M4_LOG_LEVEL=debug)."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="M4_", extra="ignore")

    service_name: str = "m4-food-recommendation"
    port: int = 8104
    log_level: str = "info"
    models_dir: str = "models"


settings = Settings()
