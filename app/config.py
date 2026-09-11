from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    log_level: str = "INFO"
    postgres_host: str = "localhost"
    postgres_port: int = 55432
    postgres_db: str = "brainseg"
    postgres_user: str = "brainseg"
    postgres_password: SecretStr = Field(default=SecretStr("brainseg"))
    postgres_schema: str = "brainseg"
    redis_url: str = "redis://localhost:56379/0"
    mlflow_tracking_uri: str = "http://localhost:55000"
    production_model_name: str = "brainseg-convnext-base"
    model_poll_seconds: int = 30
    kaggle_username: str | None = None
    kaggle_key: SecretStr | None = None
    kaggle_dataset: str = "brain-tumor-mri-dataset"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password.get_secret_value()}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def safe_database_url(self) -> str:
        return f"postgresql+psycopg://{self.postgres_user}:***@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
