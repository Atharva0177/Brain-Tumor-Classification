from typing import Any
from urllib.request import urlopen

from redis import Redis
from sqlalchemy import text

from app.config import settings
from app.db.session import engine
from app.observability import get_logger

logger = get_logger("health")


def _check_postgres() -> tuple[str, str | None]:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return "ok", None
    except Exception as exc:  # noqa: BLE001
        logger.warning("dependency_unavailable dependency=postgres detail=%s", exc)
        return "unavailable", str(exc)


def _check_redis() -> tuple[str, str | None]:
    try:
        Redis.from_url(settings.redis_url, socket_connect_timeout=1).ping()
        return "ok", None
    except Exception as exc:  # noqa: BLE001
        logger.warning("dependency_unavailable dependency=redis detail=%s", exc)
        return "unavailable", str(exc)


def _check_mlflow() -> tuple[str, str | None]:
    try:
        with urlopen(f"{settings.mlflow_tracking_uri.rstrip('/')}/health", timeout=2) as response:
            if response.status >= 400:
                return "unavailable", f"HTTP {response.status}"
        return "ok", None
    except Exception as exc:  # noqa: BLE001
        logger.warning("dependency_unavailable dependency=mlflow detail=%s", exc)
        return "unavailable", str(exc)


def dependency_health() -> dict[str, Any]:
    checks = {}
    for name, checker in (("postgres", _check_postgres), ("redis", _check_redis), ("mlflow", _check_mlflow)):
        status, detail = checker()
        checks[name] = {"status": status, **({"detail": detail} if detail else {})}
    ready = all(item["status"] == "ok" for item in checks.values())
    return {"status": "ok" if ready else "unavailable", "checks": checks}
