from pydantic import SecretStr

from app.config import Settings


def test_database_url_contains_no_unescaped_secret_in_safe_url() -> None:
    config = Settings(postgres_password=SecretStr("do-not-log"))

    assert "do-not-log" in config.database_url
    assert "do-not-log" not in config.safe_database_url
    assert "***" in config.safe_database_url


def test_defaults_match_local_compose_contract() -> None:
    config = Settings()

    assert config.postgres_host == "localhost"
    assert config.postgres_db == "brainseg"
    assert config.redis_url == "redis://localhost:56379/0"
